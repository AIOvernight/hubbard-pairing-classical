#!/usr/bin/env python3
"""Run the connected-cumulant hierarchy for one lattice and boundary sector.

Examples
--------
    # 4x2, all nine models, pure complex128 on CPU (about a minute)
    python scripts/run_hierarchy.py --geometry 4x2 --boundary PP --rank 4,5,6 --cutoff 4,5,6 \
        --device cpu --out results_4x2_pp.json

    # 6x6 production arithmetic on GPU, one sector, c4
    python scripts/run_hierarchy.py --geometry 6x6 --boundary PP --rank 4,6 --cutoff 4 \
        --device gpu --out results_6x6_pp_c4.json

    # 6x6 with six-site supports; shared six-site sources are stored in --cache-dir
    python scripts/run_hierarchy.py --geometry 6x6 --boundary PP --rank 6 --cutoff 6 \
        --device gpu --cache-dir /path/to/cache --out results_6x6_pp_r6c6.json

Cost measurements and estimates (RTX 5090, CuPy, mixed)
-------------------------------------------------------
* 4x2, one sector, r4..6 x c4..6: ~2-18 s GPU; ~1 min CPU (complex128).
* 4x4 PP, r4..6, c4: ~11 s GPU; ~20 min CPU with 6 workers.
* 6x6 PP, r4..6, c4 (all 7423 orbits): 102 s GPU wall, host RAM < 2 GB.
* 6x6 with c6: 860 six-site PP templates (+4 twisted ones per twisted
  sector) are built serially on the CPU (about a minute; 13 s when the same
  function was run with 4 processes); with ``--cache-dir`` they are stored
  once (~1.3 GB on disk) and reused by all sectors.  The full nine-model
  c<=6 sector needs several GB of GPU memory for source factors.  Runtime
  estimate: the original implementation of the same kernels took 115 min
  for all four sectors (~29 min per sector).

``--compare`` takes a previously stored result (same JSON layout) and reports
total and per-region differences for the common models.
"""
import argparse
import json
import math
import os
import pathlib
import sys
import time

os.environ.setdefault('NVIDIA_TF32_OVERRIDE', '0')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hubbard259._geometry import BOUNDARIES  # noqa: E402
from hubbard259.production import run_hierarchy  # noqa: E402


def int_list(text):
    return sorted({int(x) for x in text.split(',') if x})


def compare(result, reference_path):
    ref = json.loads(pathlib.Path(reference_path).read_text())
    ref_rows = {tuple(r['region']): r for r in ref['rows']}
    report = {}
    for key in result['rows'][0]['values'] if result['rows'] else []:
        if key not in ref_rows[next(iter(ref_rows))]['values']:
            continue
        region_max = max(abs(complex(*row['values'][key]) - complex(*ref_rows[tuple(row['region'])]['values'][key]))
                         for row in result['rows'])
        weighted = math.fsum(row['multiplicity'] * abs(row['values'][key][0] - ref_rows[tuple(row['region'])]['values'][key][0])
                             for row in result['rows'])
        entry = dict(region_max_abs=region_max, weighted_abs_sum=weighted)
        if result['totals'] and ref.get('totals'):
            entry.update(total=result['totals'][key][0], reference_total=ref['totals'][key][0],
                         total_diff=result['totals'][key][0] - ref['totals'][key][0])
        report[key] = entry
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--geometry', choices=['6x6', '4x4', '4x2'], required=True)
    p.add_argument('--boundary', choices=list(BOUNDARIES), required=True)
    p.add_argument('--rank', type=int_list, default=[4, 5, 6], help='closure ranks, e.g. 4,5,6 (each 4..6)')
    p.add_argument('--cutoff', type=int_list, default=[4], help='support cutoffs, e.g. 4,5,6 (each 4..6)')
    p.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    p.add_argument('--precision', choices=['mixed', 'double'], default=None,
                   help='GPU arithmetic (default mixed); CPU is always double')
    p.add_argument('--cache-dir', type=pathlib.Path, default=None,
                   help='directory for shared six-site source factors (c6 only)')
    p.add_argument('--batch', type=int, default=8, help='readout regions per GPU batch')
    p.add_argument('--cluster-batch', type=int, default=128, help='source clusters per GPU group')
    p.add_argument('--limit', type=int, default=0, help='evenly spread regions per support size (pilot)')
    p.add_argument('--workers', type=int, default=1, help='CPU worker processes')
    p.add_argument('--compare', type=pathlib.Path, default=None, help='stored result to compare with')
    p.add_argument('--out', type=pathlib.Path, required=True)
    a = p.parse_args()
    if not all(4 <= v <= 6 for v in a.rank + a.cutoff):
        p.error('ranks and cutoffs must lie in 4..6')
    lx, ly = map(int, a.geometry.split('x'))
    bx, by = BOUNDARIES[a.boundary]
    start = time.perf_counter()
    result = run_hierarchy(lx, ly, bx, by, a.rank, a.cutoff, a.device, a.precision, a.batch, a.cluster_batch,
                           a.limit, a.cache_dir, progress=lambda d: print(json.dumps(d), flush=True),
                           workers=a.workers)
    result['boundary_label'] = a.boundary
    result['wall_seconds'] = time.perf_counter() - start
    if a.device == 'gpu':
        import cupy as cp
        result['gpu'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        result['tf32_override'] = os.environ.get('NVIDIA_TF32_OVERRIDE')
    if a.compare:
        result['comparison'] = compare(result, a.compare)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    summary = {k: v for k, v in result.items() if k != 'rows'}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
