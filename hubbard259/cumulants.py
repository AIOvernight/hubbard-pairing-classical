"""Local raw moments and fermionic connected cumulants of preparation clusters.

Convention: for ordered mode sets A, B (bitmasks)

    Gamma_r(A, B) = < c^dag_{A ascending} c_{B descending} >,   |A| = |B| = r.

The moment-cumulant relation is evaluated recursively by distinguishing the
lowest creator a0 of A (this makes every partition appear exactly once):

    Gamma(A,B) = sum_{I subset A, a0 in I; J subset B; |I|=|J|}
                 (-1)^{shuffle(A,I)+shuffle(B,J)} kappa(I,J) Gamma(A\\I, B\\J).

``inverse_moments`` solves it for kappa (rank by rank); ``forward_cumulants``
rebuilds Gamma from a (possibly truncated) cumulant list.

Each cluster state is stored as raw moment *factors*: for every rank r either
a dense Gamma_r matrix or a Gram factor Y with Gamma_r = conj(Y) Y^T (used when
Y has fewer columns than rows).  Rows are multiplied by the fermionic
embedding sign of the local modes in the global site-major order.
"""
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit, prange, types
from numba.extending import intrinsic

from .clusters import connected_supports, prepare_cluster, source_templates


def basis(m, n):
    """Sorted bitmasks of all n-subsets of m modes."""
    return np.sort(np.array([sum(1 << k for k in ks) for ks in itertools.combinations(range(m), n)], np.uint32))


@lru_cache(None)
def comb_modes(m, r):
    """Mode lists of the rank-r basis of m modes, in sorted bitmask order."""
    return np.array([[p for p in range(m) if a & (1 << p)] for a in basis(m, r)], np.int32).reshape(-1, r)


@njit(cache=True)
def _pc(x):
    v = 0
    while x:
        v += x & 1
        x >>= 1
    return v


@njit(cache=True)
def annih_vectors(state, states, remaining, subsets):
    """Rows: c_{subset ascending} |state> expressed on the ``remaining`` basis."""
    out = np.zeros((len(subsets), len(remaining)), np.complex128)
    for a in range(len(subsets)):
        mask = subsets[a]
        for i in range(len(states)):
            s = states[i]
            if s & mask != mask:
                continue
            sign = 1
            v = s
            for p in range(32):
                if mask & (1 << p):
                    if _pc(v & ((1 << p) - 1)) % 2:
                        sign = -sign
                    v ^= 1 << p
            j = np.searchsorted(remaining, v)
            out[a, j] = sign * state[i]
    return out


@intrinsic
def _pop(typingctx, x):
    sig = types.int64(x)

    def codegen(context, builder, signature, args):
        val = args[0]
        fn = builder.module.declare_intrinsic('llvm.ctpop', [val.type])
        out = builder.call(fn, [val])
        if val.type.width < 64:
            out = builder.zext(out, context.get_value_type(types.int64))
        return out
    return sig, codegen


@njit(parallel=True, cache=True)
def lower_fast(ams, bms, kappas, moments, lookups, rank):
    """Sum of all moment-cumulant terms with |I| < rank for each (A, B)."""
    out = np.empty(len(ams), np.complex128)
    for ii in prange(len(ams)):
        am = ams[ii]
        bm = bms[ii]
        low = am & -am
        rest = am ^ low
        value = 0j
        extra = rest
        while True:
            a = low | extra
            k = _pop(a)
            if k < rank:
                asign = -(k * (k - 1) // 2)
                t = a
                while t:
                    bit = t & -t
                    asign += _pop(am & (bit - 1))
                    t ^= bit
                ia = lookups[k - 1][a]
                ja = lookups[rank - k - 1][am ^ a]
                b = bm
                while b:
                    if _pop(b) == k:
                        sg = asign - k * (k - 1) // 2
                        t = b
                        while t:
                            bit = t & -t
                            sg += _pop(bm & (bit - 1))
                            t ^= bit
                        term = kappas[k - 1][ia, lookups[k - 1][b]] * moments[rank - k - 1][ja, lookups[rank - k - 1][bm ^ b]]
                        value += -term if sg % 2 else term
                    b = (b - 1) & bm
            if extra == 0:
                break
            extra = (extra - 1) & rest
        out[ii] = value
    return out


@lru_cache(None)
def _indices(m, cap):
    masks = [basis(m, r).astype(np.int64) for r in range(1, cap + 1)]
    looks = []
    pairs = []
    for a in masks:
        look = np.full(1 << m, -1, np.int32)
        look[a] = np.arange(len(a))
        looks.append(look)
        pairs.append((np.repeat(a, len(a)), np.tile(a, len(a))))
    return masks, tuple(looks), pairs


def inverse_moments(tables, m, cap):
    """Cumulants kappa_1..kappa_cap from moments ``tables`` = [1, Gamma_1, ...]."""
    _, looks, pairs = _indices(m, cap)
    kap = [tables[1]] + [np.zeros((1, 1), complex)] * (cap - 1)
    for r in range(2, cap + 1):
        am, bm = pairs[r - 1]
        kap[r - 1] = tables[r] - lower_fast(am, bm, tuple(kap), tuple(tables[1:]), looks, r).reshape(tables[r].shape)
    return kap


def forward_cumulants(kap, m, order):
    """Moments Gamma_1..Gamma_min(6,m) from cumulants truncated at rank ``order``."""
    cap = min(6, m)
    masks, looks, pairs = _indices(m, cap)
    kk = [kap[r - 1] if r <= order and r <= len(kap) else np.zeros((len(mask), len(mask)), complex)
          for r, mask in enumerate(masks, 1)]
    tables = [kk[0]] + [np.zeros((1, 1), complex)] * (cap - 1)
    for r in range(2, cap + 1):
        am, bm = pairs[r - 1]
        tables[r - 1] = kk[r - 1] + lower_fast(am, bm, tuple(kk), tuple(tables), looks, r).reshape(kk[r - 1].shape)
    return tables, looks


# ----------------------------------------------------------------------------
# Raw moment factors of cluster states.
# ----------------------------------------------------------------------------

def raw_factors(lx, ly, sites, bx, by, order=6):
    """Raw-moment factor entries of the exact cluster state on ``sites``."""
    sites = tuple(sites)
    k = len(sites)
    state = prepare_cluster(lx, ly, sites, bx, by)
    states = basis(2 * k, k).astype(np.int64)
    v = state[states]
    entries = []
    for r in range(1, min(order, 2 * k) + 1):
        masks = basis(2 * k, r).astype(np.int64)
        b = len(masks)
        if r > k:
            entries.append(dict(kind='zero', B=b))
            continue
        gram = annih_vectors(v, states, basis(2 * k, k - r).astype(np.int64), masks)
        signs = np.array([(-1) ** sum(sites[p // 2] - p // 2 for p in row) for row in comb_modes(2 * k, r)])
        gram *= signs[:, None]
        if gram.shape[1] < b:
            entries.append(dict(kind='gram', B=b, value=gram))
        else:
            entries.append(dict(kind='dense', B=b, value=np.ascontiguousarray(gram.conj() @ gram.T)))
    return entries


def _freeze(entries):
    for e in entries:
        if 'value' in e:
            e['value'].flags.writeable = False
    return entries


def _cache_path(cache_dir, lx, ly, sites, boundary):
    name = f"{lx}x{ly}_b{boundary[0]:+d}{boundary[1]:+d}_" + '-'.join(map(str, sites)) + '.npz'
    return Path(cache_dir) / name


def _load_or_build(lx, ly, sites, boundary, order, cache_dir):
    if cache_dir is None:
        return raw_factors(lx, ly, sites, *boundary, order)
    path = _cache_path(cache_dir, lx, ly, sites, boundary)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            if meta['sites'] != list(sites) or meta['boundary'] != list(boundary) or meta['order'] < order:
                raise ValueError(f'Cache entry mismatch: {path}')
            entries = []
            for r, (kind, b) in enumerate(zip(meta['kinds'], meta['B']), 1):
                if r > order:
                    break
                entries.append(dict(kind=kind, B=b) if kind == 'zero' else dict(kind=kind, B=b, value=np.ascontiguousarray(z[f'r{r}'])))
            return entries
    entries = raw_factors(lx, ly, sites, *boundary, order)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(sites=list(sites), boundary=list(boundary), order=order,
                kinds=[e['kind'] for e in entries], B=[e['B'] for e in entries])
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, metadata=np.array(json.dumps(meta)),
             **{f'r{r}': e['value'] for r, e in enumerate(entries, 1) if 'value' in e})
    tmp.replace(path)
    return entries


def build_sources(lx, ly, bx, by, maxsites, order=6, shared_from=6, cache_dir=None, progress=None):
    """Local factor records for every connected support of size <= maxsites.

    Supports smaller than ``shared_from`` are prepared directly in boundary
    (bx, by).  Larger supports share one canonical source per even-translation
    orbit (and PP sources across boundaries through the local site gauge);
    the shared arrays are embedded with source-ordered modes and per-mode
    seam/gauge phases.  ``cache_dir`` optionally stores the shared sources.

    Each record: ``dict(sites, modes, phases, entries)``.
    """
    records = []
    for sites in connected_supports(lx, ly, min(maxsites, shared_from - 1)):
        k = len(sites)
        records.append(dict(sites=tuple(sites), modes=np.array([2 * s + t for s in sites for t in (0, 1)]),
                            phases=np.ones(2 * k), entries=_freeze(raw_factors(lx, ly, sites, bx, by, order))))
    for size in range(shared_from, maxsites + 1):
        templates = source_templates(lx, ly, bx, by, size)
        for i, (sites, boundary, maps) in enumerate(templates):
            entries = _freeze(_load_or_build(lx, ly, sites, boundary, order, cache_dir))
            for mapping in maps:
                records.append(dict(sites=mapping['sites'], modes=np.asarray(mapping['modes']),
                                    phases=np.asarray(mapping['phases'], float), entries=entries))
            if progress is not None and (i % 64 == 0 or i + 1 == len(templates)):
                progress(dict(stage='sources', size=size, completed=i + 1, templates=len(templates)))
    records.sort(key=lambda r: (len(r['sites']), r['sites']))
    expected = connected_supports(lx, ly, maxsites)
    if [r['sites'] for r in records] != list(expected):
        raise RuntimeError('Incomplete source support census')
    return records


def entry_moments(entry):
    """Dense raw moment matrix of one factor entry."""
    if entry['kind'] == 'zero':
        return np.zeros((entry['B'], entry['B']), complex)
    if entry['kind'] == 'gram':
        return entry['value'].conj() @ entry['value'].T
    return np.array(entry['value'], copy=True, order='C')


_THIRD_CACHE = {}


def third_cumulant(record):
    """Exact complex128 local third cumulant kappa_3 of a source record.

    Cached by the identity of the shared factor arrays (the arrays are kept
    referenced so identities cannot be recycled).
    """
    entries = record['entries'][:3]
    key = tuple((e['kind'], id(e.get('value'))) for e in entries)
    if key not in _THIRD_CACHE:
        tables = [np.ones((1, 1), complex)] + [entry_moments(e) for e in entries]
        third = inverse_moments(tables, 2 * len(record['sites']), 3)[2]
        _THIRD_CACHE[key] = (tuple(e.get('value') for e in entries), third)
    return _THIRD_CACHE[key][1]


def clear_third_cache():
    _THIRD_CACHE.clear()
