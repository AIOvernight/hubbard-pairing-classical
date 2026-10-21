#!/usr/bin/env python3
"""Exact small-lattice values of A for all four boundary sectors.

Examples::

    python scripts/run_exact.py --lattice 4x2                  # CPU, < 1 s
    python scripts/run_exact.py --lattice 4x4 --backend gpu    # ~5 s per sector (RTX 5090)
    python scripts/run_exact.py --lattice 8x2 --backend gpu
    python scripts/run_exact.py --lattice 4x2 --backend gpu --compare-cpu

Each sector is compared with the reference values below (exact untruncated
double-precision evolution); the script exits non-zero if any sector differs
by more than ``--tolerance``.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from hubbard259.exact import CPUBackend, evolve, simulate  # noqa: E402
from hubbard259.lattice import BOUNDARIES, BOUNDARY_NAMES  # noqa: E402

#: Reference values A per boundary sector (bx, by): PP=(+,+), AP=(-,+), PA=(+,-), AA=(-,-).
REFERENCE = {
    "4x2": {"PP": 0.48510449121154775, "AP": 0.49078793651909564,
            "PA": 0.3131753546070669, "AA": 0.3446299161882184},
    "4x4": {"PP": 0.36621853881193445, "AP": 0.3840332316570225,
            "PA": 0.3840332316570227, "AA": 0.4033717901314559},
    "8x2": {"PP": 0.4867936246610776, "AP": 0.4866551547108466,
            "PA": 0.3279614186155206, "AA": 0.3278894605898670},
}


def make_backend(kind, lx, ly):
    if kind == "cpu":
        return CPUBackend(lx, ly)
    from hubbard259.exact_gpu import GPUBackend
    return GPUBackend(lx, ly)


def compare_vectors(lx, ly):
    """Full-state CPU/GPU comparison after every stage, all sectors."""
    from hubbard259.exact_gpu import GPUBackend
    worst = 0.0
    for name in BOUNDARY_NAMES:
        bx, by = BOUNDARIES[name]
        cpu, gpu = CPUBackend(lx, ly), GPUBackend(lx, ly)
        assert np.array_equal(cpu.basis.astype(np.uint32), gpu.basis.get())
        cpu_states, gpu_states = [], []
        evolve(cpu, lx, ly, bx, by, callback=lambda s: cpu_states.append(cpu.v.copy()))
        evolve(gpu, lx, ly, bx, by, callback=lambda s: gpu_states.append(gpu.state()))
        errs = [float(np.linalg.norm(a - b)) for a, b in zip(cpu_states, gpu_states)]
        worst = max(worst, *errs)
        print(f"  {name}: CPU/GPU state L2 difference per stage {['%.2e' % e for e in errs]}")
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lattice", default="4x2", help="LXxLY, e.g. 4x2, 4x4, 8x2")
    p.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--sectors", default=",".join(BOUNDARY_NAMES), help="comma-separated subset of PP,AP,PA,AA")
    p.add_argument("--tolerance", type=float, default=1e-10)
    p.add_argument("--compare-cpu", action="store_true", help="also compare full GPU and CPU states")
    p.add_argument("--out", type=pathlib.Path, help="write JSON results here")
    a = p.parse_args()
    lx, ly = (int(x) for x in a.lattice.lower().split("x"))
    key = f"{lx}x{ly}"
    start = time.perf_counter()
    rows, ok = {}, True
    for name in a.sectors.split(","):
        bx, by = BOUNDARIES[name]
        t0 = time.perf_counter()
        backend = make_backend(a.backend, lx, ly)
        r = simulate(lx, ly, bx, by, backend)
        r["seconds_including_setup"] = time.perf_counter() - t0
        del backend
        ref = REFERENCE.get(key, {}).get(name)
        r["reference"] = ref
        r["difference"] = None if ref is None else r["A"] - ref
        if ref is not None and abs(r["difference"]) > a.tolerance:
            ok = False
        rows[name] = r
        diff = "n/a" if ref is None else f"{r['difference']:+.2e}"
        print(f"{key} {name} (bx,by)=({bx:+d},{by:+d})  A={r['A']:.12f}  ref diff {diff}  "
              f"norm-1={r['norm'] - 1:+.1e}  {r['seconds_including_setup']:.2f} s", flush=True)
    out = dict(lattice=[lx, ly], backend=a.backend, sectors=rows,
               seconds=time.perf_counter() - start)
    if len(rows) == 4:
        out["mean"] = sum(r["A"] for r in rows.values()) / 4
        print(f"{key} uniform boundary mean A={out['mean']:.12f}")
    if a.compare_cpu:
        out["cpu_gpu_state_l2_max"] = compare_vectors(lx, ly)
        print(f"max CPU/GPU state difference {out['cpu_gpu_state_l2_max']:.2e}")
        ok = ok and out["cpu_gpu_state_l2_max"] < 1e-12
    print(f"total {out['seconds']:.1f} s; {'PASS' if ok else 'FAIL'}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
