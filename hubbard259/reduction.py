"""Exact reduction of the measured observable to a dressed operator after G1.

The statistic is (author-analysis convention)

    A = (2/L) * ( <D> - <eta^+ eta^-> ),   D = sum_i n_iu n_id,
    eta^- = sum_i (-1)^(x_i+y_i) c_id c_iu,

measured after ``G1 D(3) G2 D(1.5)``.  The terminal D(1.5) commutes with A and
cancels.  Pulling A back through G2 gives the quartic

    G2^dag A G2 = sum_abcd C_abcd c^dag(a,u) c^dag(b,d) c(c,d) c(d,u),
    C_abcd = -(2/L) ( P*_ab P_dc - sum_i W*_ia,u W*_ib,d W_ic,d W_id,u ),
    P_ab   = sum_i (-1)^(x_i+y_i) W_ia,u W_ib,d .

Conjugation by the central onsite layer D(3) multiplies each quartic by a
phase and by at most four spectator number factors (1 + (e^{i theta v}-1) n).
After CAR normal ordering every monomial c^dag_A c_B has particle rank <= 6
and lives on the (at most four) sites of its quartic.  This is an exact
algebraic support bound; no truncation of the strong pulses is involved.

Monomials are keyed by bitmasks ``(A, B)`` over the local modes of a region;
their value convention is Gamma(A, B) = < c^dag_{A ascending} c_{B descending} >.
"""
import itertools
from functools import lru_cache

import numpy as np

from ._geometry import CENTRAL_ONSITE, G1, G2, PULSE_DT, gate_layout, second_propagator, stagger


@lru_cache(None)
def readout_tensor(lx, ly, bx, by):
    """Coefficients C[a,b,c,d] of the observable pulled back through G2."""
    n = lx * ly
    w2 = second_propagator(lx, ly, bx, by)
    wu, wd = w2[::2, ::2], w2[1::2, 1::2]
    eta = np.array([stagger(a, lx) for a in range(n)])
    pair = np.einsum('i,ia,ib->ab', eta, wu, wd, optimize=True)
    diagonal = np.einsum('ia,ib,ic,id->abcd', wu.conj(), wd.conj(), wd, wu, optimize=True)
    out = np.ascontiguousarray(-2 / n * (np.einsum('ab,dc->abcd', pair.conj(), pair) - diagonal))
    out.flags.writeable = False
    return out


@lru_cache(None)
def normal(term):
    """CAR normal order of a product of ladder operators.

    ``term`` is a tuple of ``(mode, is_creator)``.  Returns a dict mapping
    ``(creator_mask, annihilator_mask)`` to integer coefficients, with creators
    in ascending and annihilators in descending mode order.
    """
    for i in range(len(term) - 1):
        a, ca = term[i]
        b, cb = term[i + 1]
        if ca == cb and a == b:
            return {}
        bad = (ca < cb) or (ca == cb and ((ca and a > b) or (not ca and a < b)))
        if bad:
            swapped = term[:i] + (term[i + 1], term[i]) + term[i + 2:]
            out = {key: -v for key, v in normal(swapped).items()}
            if ca == 0 and cb == 1 and a == b:
                for key, v in normal(term[:i] + term[i + 2:]).items():
                    out[key] = out.get(key, 0) + v
            return {k: v for k, v in out.items() if v}
    am = sum(1 << p for p, c in term if c)
    bm = sum(1 << p for p, c in term if not c)
    return {(am, bm): 1}


@lru_cache(None)
def dressed_terms(a, b, c, d, theta=CENTRAL_ONSITE):
    """D(theta)^dag [c^dag(a,u) c^dag(b,d) c(c,d) c(d,u)] D(theta) in normal order.

    a, b, c, d are LOCAL site indices of the readout region; the result maps
    local mode bitmasks (A, B) to complex coefficients.
    """
    delta = {}
    for p, s in [(2 * a, 1), (2 * b + 1, 1), (2 * c + 1, -1), (2 * d, -1)]:
        delta[p] = delta.get(p, 0) + s
    # Spectator number factors act on the partner spin of every net-changed mode.
    alpha = {p ^ 1: v for p, v in delta.items() if v}
    const = sum(delta.get(2 * j, 0) * delta.get(2 * j + 1, 0) for j in {a, b, c, d})
    out = {}
    items = list(alpha.items())
    for mask in range(1 << len(items)):
        term = ((2 * a, 1), (2 * b + 1, 1), (2 * c + 1, 0), (2 * d, 0))
        weight = np.exp(1j * theta * const)
        for i, (p, v) in enumerate(items):
            if mask & (1 << i):
                term += ((p, 1), (p, 0))
                weight *= np.exp(1j * theta * v) - 1
        for key, sg in normal(term).items():
            out[key] = out.get(key, 0) + sg * weight
    return out


def region_terms(lx, ly, bx, by, region):
    """All dressed monomials whose quartic has site support exactly ``region``."""
    coeff = readout_tensor(lx, ly, bx, by)
    pos = {p: i for i, p in enumerate(region)}
    terms = {}
    for a, b, c, d in itertools.product(region, repeat=4):
        if set((a, b, c, d)) != set(region):
            continue
        weight = coeff[a, b, c, d]
        if abs(weight) < 1e-16:
            continue
        for key, value in dressed_terms(pos[a], pos[b], pos[c], pos[d]).items():
            terms[key] = terms.get(key, 0) + weight * value
    return terms


def dressed_observable(lx, ly, bx, by, tol=1e-15):
    """Full dressed observable as global-mode monomials (small lattices only).

    Returns ``(creator_masks, annihilator_masks, weights)``.
    """
    coeff = readout_tensor(lx, ly, bx, by)
    out = {}
    for a, b, c, d in np.ndindex(coeff.shape):
        v = coeff[a, b, c, d]
        if abs(v) < 1e-16:
            continue
        sites = sorted({a, b, c, d})
        pos = {p: i for i, p in enumerate(sites)}
        local = [2 * p + t for p in sites for t in (0, 1)]
        for (am, bm), w in dressed_terms(pos[a], pos[b], pos[c], pos[d]).items():
            gam = sum(1 << local[i] for i in range(len(local)) if am >> i & 1)
            gbm = sum(1 << local[i] for i in range(len(local)) if bm >> i & 1)
            # Local and global orders coincide because local modes are sorted.
            out[gam, gbm] = out.get((gam, gbm), 0) + v * w
    out = {k: v for k, v in out.items() if abs(v) > tol}
    return (np.array([a for a, _ in out], np.int64), np.array([b for _, b in out], np.int64),
            np.array(list(out.values())))


# ----------------------------------------------------------------------------
# Direct Fock-space check of the reduction (small lattices, e.g. 4x2).
# ----------------------------------------------------------------------------

def _apply_layer(v, lx, ly, bx, by, k, j):
    """One Gaussian hopping layer (both rounds) on a dense Fock vector."""
    from .clusters import hop
    for _, a, b, wy, wx, cs in gate_layout(lx, ly):
        hop(v, a, b, PULSE_DT * (bx if wx else 1) * (by if wy else 1) * (k + 1j * j * cs))


def _statistic(v, lx, ly):
    n = lx * ly
    idx = np.arange(len(v))
    prob = abs(v) ** 2
    doublons = 0.
    eta = np.zeros_like(v)
    for s in range(n):
        mask = 3 << (2 * s)
        occupied = (idx & mask) == mask
        doublons += prob[occupied].sum()
        np.add.at(eta, idx[occupied] ^ mask, stagger(s, lx) * v[occupied])
    return 2 * (doublons - np.vdot(eta, eta).real) / n


def fock_reduction_check(lx, ly, bx, by):
    """Compare the dressed operator after G1 with direct Fock evolution.

    Returns ``(A_direct, A_dressed, n_monomials)``.  Uses a dense 4^L Fock vector,
    so it is restricted to L <= 8 sites.
    """
    if lx * ly > 8:
        raise ValueError('Direct Fock check is restricted to at most 8 sites')
    from .clusters import onsite, prepare_cluster
    from .cumulants import annih_vectors, basis
    n = lx * ly
    state = prepare_cluster(lx, ly, range(n), bx, by)
    _apply_layer(state, lx, ly, bx, by, *G1)        # state right after G1
    final = state.copy()
    onsite(final, n, CENTRAL_ONSITE)
    _apply_layer(final, lx, ly, bx, by, *G2)        # D(1.5) is omitted: it commutes with A
    direct = _statistic(final, lx, ly)
    am, bm, weights = dressed_observable(lx, ly, bx, by)
    states = basis(2 * n, n).astype(np.int64)
    v = state[states]
    moments = np.zeros(len(am), complex)
    for k in range(1, 7):
        take = np.flatnonzero([int(a).bit_count() == k for a in am])
        if not len(take):
            continue
        masks = np.unique(np.concatenate((am[take], bm[take])))
        vec = annih_vectors(v, states, basis(2 * n, n - k).astype(np.int64), masks)
        ia = np.searchsorted(masks, am[take])
        ib = np.searchsorted(masks, bm[take])
        moments[take] = np.einsum('ij,ij->i', vec[ia].conj(), vec[ib])
    dressed = weights @ moments
    return float(direct), complex(dressed), len(am)
