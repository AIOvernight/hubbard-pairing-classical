"""Driver of the connected-cumulant hierarchy for one lattice and boundary sector.

``run_hierarchy`` evaluates all requested (rank r, support cutoff c) models in
a single pass:

1. build local raw-moment factors for every connected preparation support of
   size <= max(c) (:func:`hubbard259.cumulants.build_sources`);
2. integer incidence weights per cutoff replace connected-subcluster
   subtraction (:func:`hubbard259.clusters.incidence_weights`);
3. for every readout-region orbit, project every source through W1, invert
   to cumulants, weight and sum (:mod:`hubbard259.projection`);
4. reconstruct moments at rank r and contract the dressed operator
   (:mod:`hubbard259.readout`);
5. total = multiplicity-weighted ``math.fsum`` over orbits.

Devices: ``gpu`` (CuPy; precision ``mixed`` reproduces the stored
production numbers, ``double`` is all-complex128) and ``cpu`` (NumPy/Numba,
complex128 throughout; intended for 4x2 and small checks).
"""
import math
import os
import time

import numpy as np

from ._geometry import first_propagator
from .clusters import incidence_weights
from .cumulants import build_sources
from .projection import PRECISIONS, project_one
from .readout import measure_region_cpu, region_orbits
from .reduction import readout_tensor


def model_keys(ranks, cutoffs):
    return [f'r{r}c{c}' for r in ranks for c in cutoffs]


def select_regions(regions, limit):
    """Evenly spread subset (including both ends) for bounded pilots."""
    if not limit or limit >= len(regions):
        return regions
    return [regions[int(i)] for i in np.linspace(0, len(regions) - 1, limit, dtype=int)]


def _weights(records, cutoffs, n_sites):
    weights = {c: incidence_weights([r['sites'] for r in records if len(r['sites']) <= c]) for c in cutoffs}
    for c in cutoffs:
        if any(sum(v for s, v in weights[c].items() if p in s) != 1 for p in range(n_sites)):
            raise RuntimeError('Incidence weights do not partition the single-site anchor')
    return weights


def run_hierarchy(lx, ly, bx, by, ranks=(4, 5, 6), cutoffs=(4, 5, 6), device='gpu', precision=None,
                  batch=8, cluster_batch=128, limit=0, cache_dir=None, regions=None, progress=None,
                  workers=1):
    """Evaluate A for all (rank, cutoff) models; returns a JSON-serialisable dict.

    ``regions``: optional explicit list of canonical regions (each must be an
    orbit representative) instead of the complete census; ``limit`` keeps an
    evenly spread subset per support size.  Totals are only reported for
    complete runs.
    """
    ranks = sorted(set(ranks))
    cutoffs = sorted(set(cutoffs))
    if not (1 <= min(ranks) and max(ranks) <= 6 and 1 <= min(cutoffs) and max(cutoffs) <= 6):
        raise ValueError('Require ranks and cutoffs in 1..6')
    if precision is None:
        precision = 'mixed' if device == 'gpu' else 'double'
    if precision not in PRECISIONS or (device == 'cpu' and precision != 'double'):
        raise ValueError('CPU mode is pure complex128; GPU supports double|mixed')
    log = progress or (lambda _: None)
    start = time.perf_counter()
    order = max(ranks)
    shared_from = 7 if lx * ly <= 8 else 6
    records = build_sources(lx, ly, bx, by, max(cutoffs), 6, shared_from, cache_dir, log)
    weights = _weights(records, cutoffs, lx * ly)
    prep_seconds = time.perf_counter() - start
    log(dict(stage='sources_ready', records=len(records), seconds=prep_seconds))
    families = {s: region_orbits(lx, ly, s) for s in range(1, 5)}
    counts = {str(s): len(f[0]) for s, f in families.items()}
    if regions is not None:
        wanted = {tuple(r) for r in regions}
    keys = model_keys(ranks, cutoffs)
    rows = []
    timings = []
    runner = _run_gpu if device == 'gpu' else _run_cpu
    for s in range(1, 5):
        allregions, mult = families[s]
        multiplicity = dict(zip(allregions, mult))
        chosen = [r for r in allregions if r in wanted] if regions is not None else select_regions(allregions, limit)
        if not chosen:
            continue
        t = time.perf_counter()
        values = runner(lx, ly, bx, by, s, chosen, records, weights, ranks, cutoffs, order, precision,
                        batch, cluster_batch, workers, log)
        timings.append(dict(support=s, regions=len(chosen), seconds=time.perf_counter() - t))
        for region, v in zip(chosen, values):
            rows.append(dict(region=list(region), multiplicity=int(multiplicity[region]),
                             values={k: [float(v[k].real), float(v[k].imag)] for k in keys}))
        log(dict(stage='support_done', support=s, regions=len(chosen), elapsed=time.perf_counter() - start))
    complete = len(rows) == sum(len(f[0]) for f in families.values())
    sums = {k: [math.fsum(row['values'][k][i] * row['multiplicity'] for row in rows) for i in (0, 1)] for k in keys}
    if complete and sum(row['multiplicity'] for row in rows) != sum(math.comb(lx * ly, s) for s in range(1, 5)):
        raise RuntimeError('Orbit multiplicities do not cover all readout supports')
    return dict(status='complete' if complete else 'partial', geometry=[lx, ly], boundary=[bx, by],
                ranks=ranks, cutoffs=cutoffs, device=device, precision=precision,
                arithmetic_policy=arithmetic_policy(precision),
                source_supports={str(k): sum(len(r['sites']) == k for r in records) for k in range(1, 7)},
                orbit_counts=counts, regions=len(rows),
                totals=sums if complete else None, partial_totals=None if complete else sums,
                prep_seconds=prep_seconds, support_timings=timings, seconds=time.perf_counter() - start,
                rows=rows)


def arithmetic_policy(precision):
    if precision == 'double':
        return dict(all='complex128')
    return dict(raw_moment_by_rank={'1': 'complex128', '2': 'complex128', '3': 'complex128_reconstruction',
                                    '4': 'complex64', '5': 'complex64', '6': 'complex64'},
                projected_kappa3='complex64', local_kappa3='complex128',
                gamma3_reconstruction='projected_kappa3_plus_lower3_complex128',
                compound='complex128', inverse='complex128', weighted_sum='complex128', closure='complex128')


# ----------------------------------------------------------------------------
# GPU path.
# ----------------------------------------------------------------------------

def _run_gpu(lx, ly, bx, by, s, regions, records, weights, ranks, cutoffs, order, precision,
             batch, cluster_batch, workers, log):
    import cupy as cp
    from cupy_backends.cuda.libs import cublas
    from .cumulants import basis
    from .projection import FactorGPU, MomentLayout
    from .readout import MomentGPU
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setMathMode(handle, 2)  # CUBLAS_PEDANTIC_MATH: no reduced-precision shortcuts
    w = cp.asarray(first_propagator(lx, ly, bx, by))
    coeff = cp.asarray(readout_tensor(lx, ly, bx, by))
    layout = MomentLayout(2 * s)
    groups = []
    initial = regions[:batch]
    for k in range(1, max(cutoffs) + 1):
        selected = [r for r in records if len(r['sites']) == k and any(weights[c].get(r['sites'], 0) for c in cutoffs)]
        for begin in range(0, len(selected), cluster_batch):
            chunk = selected[begin:begin + cluster_batch]
            device = FactorGPU(chunk, initial, w, layout, precision, order)
            ww = {c: cp.asarray(np.array([weights[c].get(r['sites'], 0) for r in chunk], np.float64)) for c in cutoffs}
            groups.append((device, ww))
    moments = {r: MomentGPU(s, r) for r in ranks}
    cap = min(order, 2 * s)
    out = []
    for begin in range(0, len(regions), batch):
        chunk = regions[begin:begin + batch]
        R = len(chunk)
        t = time.perf_counter()
        totals = {c: [cp.zeros((R, len(basis(2 * s, r)), len(basis(2 * s, r))), cp.complex128) for r in range(1, cap + 1)]
                  for c in cutoffs}
        for device, ww in groups:
            device.set_regions(chunk)
            projected = device.project()
            for c in cutoffs:
                for r, value in enumerate(projected):
                    totals[c][r] += (value * ww[c][None, :, None, None]).sum(axis=1)
            del projected
        values = {}
        for r in ranks:
            for c in cutoffs:
                gamma = moments[r].moments(totals[c])
                if not bool(cp.isfinite(gamma).all()):
                    raise FloatingPointError('Nonfinite reconstructed moment')
                values[f'r{r}c{c}'] = cp.asnumpy(moments[r].contract(gamma, chunk, coeff))
        cp.cuda.Stream.null.synchronize()
        out.extend({k: v[i] for k, v in values.items()} for i in range(R))
        if begin % (64 * batch) == 0 or begin + R == len(regions):
            log(dict(stage='regions', support=s, completed=begin + R, total=len(regions),
                     last_batch_seconds=time.perf_counter() - t))
    del groups, totals
    cp.get_default_memory_pool().free_all_blocks()
    return out


# ----------------------------------------------------------------------------
# CPU path (complex128 throughout).
# ----------------------------------------------------------------------------

_CPU_STATE = {}


def _cpu_region(region):
    st = _CPU_STATE
    lx, ly, bx, by = st['geometry']
    output = [2 * p + t for p in region for t in (0, 1)]
    m = len(output)
    cap = min(m, st['order'])
    from .cumulants import basis
    totals = {c: [np.zeros((len(basis(m, r)), len(basis(m, r))), complex) for r in range(1, cap + 1)] for c in st['cutoffs']}
    w = st['w']
    for record in st['records']:
        factors = {c: st['weights'][c].get(record['sites'], 0) for c in st['cutoffs']}
        if not any(factors.values()):
            continue
        if not np.any(w[np.ix_(output, record['modes'])]):
            continue  # outside the light cone of W1: every compound vanishes exactly
        projected = project_one(record, w, output, st['order'])
        for c, f in factors.items():
            if f:
                for r, a in enumerate(projected):
                    totals[c][r] += f * a
    return {f'r{r}c{c}': measure_region_cpu(totals[c], region, r, lx, ly, bx, by)
            for r in st['ranks'] for c in st['cutoffs']}


def _run_cpu(lx, ly, bx, by, s, regions, records, weights, ranks, cutoffs, order, precision,
             batch, cluster_batch, workers, log):
    _CPU_STATE.update(geometry=(lx, ly, bx, by), records=records, weights=weights, ranks=ranks,
                      cutoffs=cutoffs, order=order, w=first_propagator(lx, ly, bx, by))
    if workers <= 1:
        return [_cpu_region(r) for r in regions]
    import multiprocessing
    ctx = multiprocessing.get_context('fork')
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    with ctx.Pool(workers) as pool:
        return pool.map(_cpu_region, regions, chunksize=max(1, len(regions) // (8 * workers)))
