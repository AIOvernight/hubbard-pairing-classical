#!/usr/bin/env python3
"""Verify the stored results: hashes, means, the error envelope and CSV/JSON consistency.

    python verify.py [--results DIR]

CPU only, standard library only, a few seconds.  Every quantity is recomputed
independently of the package code; the package error module is additionally
cross-checked when it is importable.  Prints one line per check and a final
PASS/FAIL summary; the exit code is 0 only if every check passes.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SECTORS = ("PP", "AP", "PA", "AA")
SIGNS = {"PP": (1, 1), "AP": (-1, 1), "PA": (1, -1), "AA": (-1, -1)}
SETTINGS = [(r, c) for r in (4, 5, 6) for c in (4, 5, 6)]

# Published headline values (rounded as quoted in the text).
HEADLINE_A = "0.3836428102"
HEADLINE_SECTORS = {"PP": "0.3832431205", "AP": "0.3836285668", "PA": "0.3836290019", "AA": "0.3840705515"}
HEADLINE_ENVELOPE = "0.0049704"
ENVELOPE_FULL = 0.004970377382652378
NUMERICAL_ALLOWANCE = 0.002
TAU = 4

results_log = []


def check(name, ok, detail=""):
    results_log.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


def fmean(xs):
    xs = list(xs)
    return math.fsum(xs) / len(xs)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(s):
    return None if s == "" else (int(s) if s.lstrip("-").isdigit() else float(s))


def same_rows(csv_rows, json_rows, fields):
    if len(csv_rows) != len(json_rows):
        return False
    for c, j in zip(csv_rows, json_rows):
        for k in fields:
            jv = j[k]
            cv = c[k] if isinstance(jv, str) else num(c[k])
            if cv != jv:
                return False
    return True


def check_hashes(res):
    sums = res / "SHA256SUMS"
    if not check("SHA256SUMS present", sums.exists()):
        return
    listed, bad = {}, []
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        listed[rel] = digest
        p = res / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            bad.append(rel)
    present = {p.relative_to(res).as_posix() for p in res.rglob("*")
               if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts}
    check("SHA256SUMS: all listed files match", not bad, f"{len(listed)} files" + (f"; bad: {bad}" if bad else ""))
    check("SHA256SUMS: no unlisted result files", present == set(listed),
          ", ".join(sorted(present ^ set(listed))) or "complete")


def check_6x6(res):
    h = load(res / "hierarchy_6x6.json")
    recs = h["records"]
    grid = {(r["boundary"], r["rank"], r["cutoff"]): r for r in recs}
    check("6x6: 4 sectors x 9 settings, unique", len(recs) == 36 and len(grid) == 36
          and set(grid) == {(b, r, c) for b in SECTORS for r, c in SETTINGS})
    check("6x6: boundary signs", all((r["bx"], r["by"]) == SIGNS[r["boundary"]] for r in recs))
    check("6x6: readout coverage 7423 orbits / 66711 supports per model",
          all(r["readout_orbits"] == 7423 and r["readout_supports"] == 66711 for r in recs))
    max_imag = max(abs(r["A_imag"]) for r in recs)
    check("6x6: imaginary totals <= 1e-7", max_imag <= 1e-7, f"max {max_imag:.3e}")
    means = {(m["rank"], m["cutoff"]): m["A_mean"] for m in h["uniform_means"]}
    recomputed = {(r, c): fmean(grid[(b, r, c)]["A"] for b in SECTORS) for r, c in SETTINGS}
    check("6x6: 9 uniform means recomputed exactly", means == recomputed)
    a = recomputed[(6, 6)]
    check("6x6: reported r6c6 mean", h["reported_setting"]["A_mean"] == a and f"{a:.10f}" == HEADLINE_A,
          repr(a))
    check("6x6: per-sector r6c6 values match quoted digits",
          all(f"{grid[(b, 6, 6)]['A']:.10f}" == HEADLINE_SECTORS[b] for b in SECTORS))
    rows = read_csv(res / "hierarchy_6x6.csv")
    fields = ["boundary", "bx", "by", "rank", "cutoff", "A", "A_imag", "readout_orbits", "readout_supports"]
    data = [r for r in rows if r["boundary"] != "mean"]
    mrows = [r for r in rows if r["boundary"] == "mean"]
    check("6x6: CSV records == JSON records", same_rows(data, recs, fields))
    check("6x6: CSV means == recomputed means", len(mrows) == 9 and all(
        float(m["A"]) == recomputed[(int(m["rank"]), int(m["cutoff"]))] for m in mrows))
    return grid


def check_small(res):
    ex = load(res / "exact_small.json")
    exact = {(r["lattice"], r["boundary"]): r["A"] for r in ex["records"]}
    check("exact: 4x2, 4x4, 8x2 x 4 sectors", set(exact) == {(l, b) for l in ("4x2", "4x4", "8x2") for b in SECTORS})
    em = {m["lattice"]: m["A_mean"] for m in ex["uniform_means"]}
    check("exact: uniform means recomputed exactly",
          em == {l: fmean(exact[(l, b)] for b in SECTORS) for l in ("4x2", "4x4", "8x2")},
          ", ".join(f"{l} {v:.10f}" for l, v in em.items()))
    rows = read_csv(res / "exact_small.csv")
    check("exact: CSV == JSON", same_rows([r for r in rows if r["boundary"] != "mean"], ex["records"],
                                         ["lattice", "boundary", "bx", "by", "A"])
          and all(float(r["A"]) == em[r["lattice"]] for r in rows if r["boundary"] == "mean"))

    hs = load(res / "hierarchy_small.json")
    recs = hs["records"]
    check("small hierarchy: exact column equals exact_small",
          all(r["exact"] == exact[(r["lattice"], r["boundary"])] for r in recs))
    check("small hierarchy: residual = A - exact", all(abs(r["residual"] - (r["A"] - r["exact"])) <= 1e-15 for r in recs))
    cal = [r for r in recs if r["role"] == "calibration"]
    check("small hierarchy: 4x2/4x4 complete (72 models)", len(cal) == 72 and {
        (r["lattice"], r["boundary"], r["rank"], r["cutoff"]) for r in cal} ==
        {(l, b, r, c) for l in ("4x2", "4x4") for b in SECTORS for r, c in SETTINGS})
    groups = {}
    for r in recs:
        if r["role"] != "holdout_proxy":
            groups.setdefault((r["lattice"], r["rank"], r["cutoff"]), []).append(r)
    ok = len(groups) == len(hs["uniform_means"])
    for m in hs["uniform_means"]:
        g = groups.get((m["lattice"], m["rank"], m["cutoff"]), [])
        ok &= [r["boundary"] for r in g] == list(SECTORS)
        ok &= m["A_mean"] == fmean(r["A"] for r in g) and m["exact_mean"] == fmean(r["exact"] for r in g)
    check("small hierarchy: uniform means recomputed exactly", ok, f"{len(groups)} settings")
    rows = read_csv(res / "hierarchy_small.csv")
    check("small hierarchy: CSV == JSON", same_rows(rows, recs, ["lattice", "boundary", "bx", "by", "rank", "cutoff",
                                                                  "A", "A_imag", "exact", "residual", "arithmetic", "role"]))
    return recs


def check_envelope(res, grid, small):
    idx = {(r["lattice"], r["boundary"], r["rank"], r["cutoff"], r["arithmetic"]): r for r in small}
    table = [(l, b, idx[(l, b, 6, 6, "mixed")]["A"] - idx[(l, b, 6, 6, "mixed")]["exact"])
             for l in ("4x2", "4x4") for b in SECTORS]
    R = max(abs(t[2]) for t in table)
    worst = max(table, key=lambda t: abs(t[2]))
    cal = load(res / "calibration.json")
    check("calibration: R recomputed from 8 matched sectors", cal["R"] == R
          and (cal["worst"]["lattice"], cal["worst"]["boundary"]) == worst[:2], f"R = {R!r} ({worst[0]} {worst[1]})")
    crow = read_csv(res / "calibration.csv")
    check("calibration: CSV == JSON == recomputed", len(crow) == 8 and all(
        (c["lattice"], c["boundary"], float(c["residual"])) == t and float(c["abs_residual"]) == abs(t[2])
        for c, t in zip(crow, table)) and same_rows(crow, cal["rows"], list(crow[0])))

    B, d = {}, {}
    for b in SECTORS:
        a4, a5, a6 = (grid[(b, 6, c)]["A"] for c in (4, 5, 6))
        d[b] = (a5 - a4, a6 - a5)
        B[b] = max(R, abs(d[b][0]), abs(d[b][1]))
    mean_B = fmean(B.values())
    total = TAU * mean_B + (0.0 + NUMERICAL_ALLOWANCE)
    check("envelope: E_tot = 4*mean(B) + 0 + 0.002 recomputed", total == ENVELOPE_FULL
          and f"{total:.7f}" == HEADLINE_ENVELOPE, f"{total!r}")
    eb = load(res / "error_budget.json")
    check("envelope: error_budget.json agrees", eb["reported_envelope"] == total
          and eb["calibration"]["R"] == R and eb["estimate"]["A"] == fmean(grid[(b, 6, 6)]["A"] for b in SECTORS)
          and all(s["B"] == B[s["boundary"]] and (s["d45"], s["d56"]) == d[s["boundary"]] for s in eb["sectors"])
          and all(eb["envelopes"][str(t)]["total"] == t * mean_B + NUMERICAL_ALLOWANCE for t in (1, 2, 4)))
    tau_max = (0.02 - NUMERICAL_ALLOWANCE) / mean_B
    check("envelope: tau at which E_tot reaches 0.02", eb["tau_max_for_tolerance"]["tau"] == tau_max, f"{tau_max:.6f}")

    ok = True
    for fold in eb["holdout_checks"]:
        tr, ho = fold["training_lattice"], fold["heldout_lattice"]
        Rt = max(abs(idx[(tr, b, 6, 5, "fp64")]["residual"]) for b in SECTORS)
        ok &= fold["training_R"] == Rt
        for row in fold["rows"]:
            b = row["boundary"]
            Bh = max(Rt, abs(idx[(ho, b, 6, 5, "fp64")]["A"] - idx[(ho, b, 6, 4, "fp64")]["A"]))
            res6 = idx[(ho, b, 6, 6, "mixed")]["residual"]
            ok &= row["B"] == Bh and row["residual_r6c6"] == res6 and abs(res6) <= Bh
    check("hold-out: 8 leave-one-lattice-out cases pass already at tau = 1", ok)
    ok = True
    for b in SECTORS:
        a = [idx[("8x2", b, 6, c, "fp64")]["A"] for c in (4, 5, 6)]
        Bv = max(R, abs(a[1] - a[0]), abs(a[2] - a[1]))
        ok &= abs(a[2] - idx[("8x2", b, 6, 6, "fp64")]["exact"]) <= TAU * Bv
    check("8x2 validation: all sectors within the H(4) proxy", ok)
    st = eb["parameter_stress_4x2"]["rows"]
    check("parameter stress table: signed error = r6c6 mean - exact mean",
          all(abs(r[3] - (r[2] - r[1])) < 1e-11 for r in st), f"{len(st)} rows")
    try:
        sys.path.insert(0, str(ROOT))
        from hubbard259 import error
    except Exception as exc:  # pragma: no cover - package optional for this check
        check("package error module cross-check (skipped)", True, f"import failed: {exc}")
        return
    rebuilt = error.error_budget(load(res / "hierarchy_6x6.json"), load(res / "hierarchy_small.json"))
    check("package error module reproduces error_budget.json", json.loads(json.dumps(rebuilt)) == eb)
    check("package error module reproduces calibration.json",
          json.loads(json.dumps(error.calibration_report(load(res / "hierarchy_small.json")))) == cal)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    try:
        check_hashes(args.results)
        grid = check_6x6(args.results)
        small = check_small(args.results)
        check_envelope(args.results, grid, small)
    except Exception as exc:
        check("verification completed without error", False, f"{type(exc).__name__}: {exc}")
    n_fail = sum(not ok for _, ok in results_log)
    print("-" * 72)
    print(f"{'PASS' if n_fail == 0 else 'FAIL'}: {len(results_log) - n_fail}/{len(results_log)} checks passed "
          f"in {time.perf_counter() - t0:.2f} s")
    if n_fail == 0:
        print(f"A = {HEADLINE_A} (6x6, rank 6, cutoff 6, uniform mean of four boundary sectors)")
        print(f"conditional envelope +-{HEADLINE_ENVELOPE} = 4*R + 0.002 under H(4) (not a rigorous bound)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
