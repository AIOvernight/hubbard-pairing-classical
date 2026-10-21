"""Connected preparation clusters: exact local Fock states and support bookkeeping.

For a connected active site set T the preparation is evolved exactly in the
local Fock space of its 2|T| modes:

* the X-Neel spinors ``[1, -(-1)^(x+y)]/sqrt(2)`` and the four spin layers are
  applied on T (two-site spin gates only if both sites lie in T);
* every weak hopping gate whose endpoints both lie in T is retained, including
  the boundary sign; omitted intermediate sites are singly occupied, which
  fixes the fermionic string sign of the local hop;
* local onsite layers, the staggered spin hop and the final onsite layer are
  applied exactly;
* spectators remain fixed singly occupied spinors; their global phase is
  included.

No global Fock vector is built.  The module also provides the connected
support census, even-translation/boundary-gauge maps used to share local
factors between translated clusters, and the integer incidence weights that
replace explicit subtraction of connected subclusters.
"""
import itertools
import math
from functools import lru_cache

import numpy as np
from numba import njit

from ._geometry import (PREP_FINAL_ONSITE, PREP_ONSITE, SPIN_HOP, SPIN_LAYERS, WEAK_HOP,
                        gate_layout, spin_bonds, stagger)


# ----------------------------------------------------------------------------
# Elementary Fock-space gates (dense vector over 2^(2k) site-major modes).
# ----------------------------------------------------------------------------

@njit(cache=True)
def _parity(x):
    p = 0
    while x:
        p ^= 1
        x &= x - 1
    return p


@njit(cache=True)
def hop(v, a, b, z):
    """exp(-i (z c^dag_a c_b + h.c.)) on a dense Fock vector (in place)."""
    mag = abs(z)
    if mag == 0:
        return
    ma, mb = 1 << a, 1 << b
    between = (1 << max(a, b)) - (1 << (min(a, b) + 1))
    c = math.cos(mag)
    f = math.sin(mag) * z / mag
    for i in range(len(v)):
        if not (i & ma) and i & mb:
            j = i ^ ma ^ mb
            sg = 1 - 2 * _parity(i & between)
            x, y = v[i], v[j]
            v[i] = c * x - 1j * np.conj(f) * sg * y
            v[j] = c * y - 1j * f * sg * x


@njit(cache=True)
def onsite(v, n, theta):
    """Onsite layer exp(-i theta/4 sum_s Z_s Z'_s) on n sites (in place)."""
    for i in range(len(v)):
        zz = 0
        for s in range(n):
            zz += 1 - 2 * (((i >> (2 * s)) ^ (i >> (2 * s + 1))) & 1)
        v[i] *= np.exp(-0.25j * theta * zz)


# ----------------------------------------------------------------------------
# Local cluster preparation.
# ----------------------------------------------------------------------------

def prepare_cluster(lx, ly, sites, bx=1, by=1):
    """Exact prepared state of active sites ``sites`` as a dense 2^(2k) vector.

    Local mode ``2*i + spin`` refers to the i-th smallest active site.
    """
    sites = tuple(sorted(sites))
    k = len(sites)
    pos = {s: i for i, s in enumerate(sites)}
    ids = np.arange(1 << k)
    spin = np.ones(1 << k, complex) / math.sqrt(1 << k)
    for i, s in enumerate(sites):
        if (s % lx + s // lx) % 2 == 0:
            spin *= 1 - 2 * ((ids >> i) & 1)
    for axis, t in SPIN_LAYERS:
        if axis == 'X':
            for i, s in enumerate(sites):
                ii = ids[(ids & (1 << i)) == 0]
                jj = ii ^ (1 << i)
                x, y = spin[ii].copy(), spin[jj].copy()
                z = -1j * math.sin(t) * stagger(s, lx)
                spin[ii] = math.cos(t) * x + z * y
                spin[jj] = math.cos(t) * y + z * x
        else:
            for a, b in spin_bonds(lx, ly):
                if a not in pos or b not in pos:
                    continue
                ia, ib = pos[a], pos[b]
                zz = (1 - 2 * ((ids >> ia) & 1)) * (1 - 2 * ((ids >> ib) & 1))
                if axis == 'Z':
                    spin *= np.exp(-1j * t * zz)
                else:
                    spin = math.cos(t) * spin + 1j * math.sin(t) * zz * spin[ids ^ (1 << ia) ^ (1 << ib)]
    occ = np.zeros(1 << k, np.int64)
    for i in range(k):
        occ |= (1 << (2 * i)) << ((ids >> i) & 1)
    state = np.zeros(1 << (2 * k), complex)
    state[occ] = spin
    for rnd in (0, 1):
        for rr, a, b, wy, wx, cs in gate_layout(lx, ly):
            sa, sb = a // 2, b // 2
            if rr != rnd or sa not in pos or sb not in pos:
                continue
            # Every omitted intermediate site carries exactly one fermion.
            spectator_parity = (-1) ** (abs(sa - sb) - abs(pos[sa] - pos[sb]))
            z = WEAK_HOP * (bx if wx else 1) * (by if wy else 1) * spectator_parity
            hop(state, 2 * pos[sa] + a % 2, 2 * pos[sb] + b % 2, z + 0j)
        onsite(state, k, PREP_ONSITE)
    for i, s in enumerate(sites):
        hop(state, 2 * i, 2 * i + 1, SPIN_HOP * stagger(s, lx) + 0j)
    onsite(state, k, PREP_FINAL_ONSITE)
    spectators = lx * ly - k
    fixed_rotation = SPIN_LAYERS[1][1] + SPIN_LAYERS[3][1] + SPIN_HOP
    fixed_onsite = 2 * PREP_ONSITE + PREP_FINAL_ONSITE
    state *= np.exp(1j * spectators * (fixed_rotation + fixed_onsite / 4))
    return state


# ----------------------------------------------------------------------------
# Support census.
# ----------------------------------------------------------------------------

def neighbors(s, lx, ly):
    x, y = s % lx, s // lx
    return [(x + dx) % lx + lx * ((y + dy) % ly) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]


@lru_cache(None)
def connected_supports(lx, ly, maximum):
    """All connected site sets of size <= maximum, sorted by (size, sites)."""
    layer = {(s,) for s in range(lx * ly)}
    out = sorted(layer)
    for _ in range(2, maximum + 1):
        layer = {tuple(sorted((*sites, q))) for sites in layer
                 for s in sites for q in neighbors(s, lx, ly) if q not in sites}
        out.extend(sorted(layer))
    return tuple(out)


def incidence_weights(supports):
    """Integer weights w(S) with sum_S Delta(S) = sum_S w(S) F(S).

    Delta(S) = F(S) - sum_{T proper connected subset of S} Delta(T) is the
    connected-subcluster subtraction; applied to local cumulants F(S) of the
    cluster states it is equivalent to the explicit recursion.  Every site has
    total weight one, so the single-site Gaussian anchor cancels exactly.
    """
    family = sorted({tuple(sorted(s)) for s in supports}, key=lambda s: (len(s), s))
    weights = {s: 1 for s in family}
    for s in reversed(family):
        value = weights[s]
        for k in range(1, len(s)):
            for t in itertools.combinations(s, k):
                if t in weights:
                    weights[t] -= value
    return weights


# ----------------------------------------------------------------------------
# Sharing local factors between translated / gauge-related clusters.
# ----------------------------------------------------------------------------

def _translated(sites, lx, ly, dx, dy):
    if dx % 2 or dy % 2:
        raise ValueError('Only even translations preserve the preparation schedule')
    return [(s % lx + dx) % lx + lx * ((s // lx + dy) % ly) for s in sites]


def boundary_site_gauge(sites, lx, ly, bx, by):
    """Site signs eta with eta_a eta_b = boundary sign of every active bond.

    Raises ValueError when the active support winds around a twisted torus
    direction (nontrivial holonomy); such clusters need their own source.
    """
    support = set(sites)
    eta = {}
    for root in sorted(support):
        if root in eta:
            continue
        eta[root] = 1
        stack = [root]
        while stack:
            a = stack.pop()
            x, y = a % lx, a // lx
            for ex, ey in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                xx, yy = x + ex, y + ey
                b = xx % lx + lx * (yy % ly)
                if b not in support:
                    continue
                sign = (bx if xx < 0 or xx >= lx else 1) * (by if yy < 0 or yy >= ly else 1)
                value = eta[a] * sign
                if b in eta:
                    if eta[b] != value:
                        raise ValueError('Nontrivial boundary holonomy on active support')
                else:
                    eta[b] = value
                    stack.append(b)
    return eta


def gaugeable(sites, lx, ly, bx, by):
    try:
        boundary_site_gauge(sites, lx, ly, bx, by)
        return True
    except ValueError:
        return False


def translation_map(sites, lx, ly, dx, dy, bx=1, by=1):
    """Same-boundary source -> translated support (source-ordered modes, seam phases)."""
    moved = _translated(sites, lx, ly, dx, dy)
    phase = [bx ** ((s % lx + dx) // lx) * by ** ((s // lx + dy) // ly) for s in sites]
    return dict(sites=tuple(sorted(moved)), modes=tuple(2 * s + t for s in moved for t in (0, 1)),
                phases=tuple(p for p in phase for _ in (0, 1)))


def pp_gauge_map(sites, lx, ly, dx, dy, bx=1, by=1):
    """PP source -> translated support of boundary (bx, by) via the local site gauge."""
    moved = _translated(sites, lx, ly, dx, dy)
    eta = boundary_site_gauge(moved, lx, ly, bx, by)
    return dict(sites=tuple(sorted(moved)), modes=tuple(2 * s + t for s in moved for t in (0, 1)),
                phases=tuple(eta[s] for s in moved for _ in (0, 1)))


def even_shifts(lx, ly):
    return list(itertools.product(range(0, lx, 2), range(0, ly, 2)))


def canonical_support(sites, lx, ly):
    return min(translation_map(sites, lx, ly, x, y)['sites'] for x, y in even_shifts(lx, ly))


def source_templates(lx, ly, bx, by, size):
    """Canonical sources for all connected supports of one size.

    Returns a list of ``(source_sites, source_boundary, [mapping, ...])``: a PP
    source is reused through the local gauge when the destination does not
    wind around a twisted direction; otherwise a source evaluated directly in
    the target boundary is translated with seam phases.
    """
    family = [s for s in connected_supports(lx, ly, size) if len(s) == size]
    reps = sorted({canonical_support(s, lx, ly) for s in family})
    covered = set()
    out = []
    for s in reps:
        for source_boundary in [(1, 1), (bx, by)] if (bx, by) != (1, 1) else [(1, 1)]:
            maps = []
            for dx, dy in even_shifts(lx, ly):
                if source_boundary == (1, 1):
                    try:
                        mapping = pp_gauge_map(s, lx, ly, dx, dy, bx, by)
                    except ValueError:
                        continue
                else:
                    mapping = translation_map(s, lx, ly, dx, dy, bx, by)
                if mapping['sites'] in covered:
                    continue
                covered.add(mapping['sites'])
                maps.append(mapping)
            if maps:
                out.append((s, source_boundary, maps))
    if covered != set(family):
        raise RuntimeError('Incomplete source census')
    return out
