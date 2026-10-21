#!/usr/bin/env python3
"""Recompute the calibration table and the conditional error budget from results/.

Reads results/hierarchy_6x6.json and results/hierarchy_small.json and writes
results/calibration.json, results/calibration.csv and results/error_budget.json.

    python scripts/estimate_error.py [--results DIR] [--check]

With --check nothing is written; the recomputed files are compared with the
stored ones and the exit code is non-zero on any difference.
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hubbard259 import error  # noqa: E402

CAL_FIELDS = ["lattice", "boundary", "A", "exact", "residual", "abs_residual"]


def render(results: Path) -> dict:
    h66 = error.load_json(results / "hierarchy_6x6.json")
    small = error.load_json(results / "hierarchy_small.json")
    cal = error.calibration_report(small)
    budget = error.error_budget(h66, small)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CAL_FIELDS)
    for r in cal["rows"]:
        w.writerow([repr(r[k]) if isinstance(r[k], float) else r[k] for k in CAL_FIELDS])
    return {
        "calibration.json": json.dumps(cal, indent=1) + "\n",
        "calibration.csv": buf.getvalue(),
        "error_budget.json": json.dumps(budget, indent=1) + "\n",
    }, budget


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=error.results_dir())
    ap.add_argument("--check", action="store_true", help="compare with stored files instead of writing")
    args = ap.parse_args(argv)
    files, budget = render(args.results)
    bad = 0
    for name, text in files.items():
        path = args.results / name
        if args.check:
            same = path.exists() and path.read_text(encoding="utf-8") == text
            print(f"{'ok      ' if same else 'MISMATCH'} {name}")
            bad += not same
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}")
    e = budget["envelopes"][str(error.TAU_REPORTED)]
    print(f"A(r6c6 mean) = {budget['estimate']['A']!r}")
    print(f"R = {budget['calibration']['R']!r} ({budget['calibration']['worst_lattice']} "
          f"{budget['calibration']['worst_boundary']})")
    print(f"E_tot(tau=4) = 4*{e['mean_B']!r} + {e['pruning_allowance']!r} + {e['numerical_allowance']!r} "
          f"= {e['total']!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
