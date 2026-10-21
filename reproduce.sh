#!/usr/bin/env bash
# Reproduction tiers for the two-step 6x6 Hubbard pairing estimate.
#
#   ./reproduce.sh verify   CPU: unit tests + consistency of stored results
#   ./reproduce.sh cheap    exact 4x2 (CPU) and 4x4 (GPU), one 6x6 sector at cutoff 4 (GPU),
#                           each compared with results/
#   ./reproduce.sh full     full 6x6 production, all four boundaries through r6c6
#                           (original computation: 115 min on RTX 5090; ~1.3 GB cache on disk)
#
# Environment:
#   PYTHON       Python interpreter (default: python3)
#   OUT          output directory for regenerated results (default: runs/<tier>)
#   CACHE_DIR    cache directory for the full production (default: cache/)
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
TIER="${1:-}"
export PYTHONDONTWRITEBYTECODE=1
# Deterministic, reproducible GPU arithmetic (no TF32 in matrix products).
export NVIDIA_TF32_OVERRIDE=0

usage() {
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

need_gpu() {
    if ! "$PYTHON" -c "import cupy; cupy.cuda.runtime.getDeviceCount()" >/dev/null 2>&1; then
        echo "error: this tier needs CuPy and an NVIDIA GPU (pip install -r requirements-gpu.txt)" >&2
        exit 2
    fi
}

stamp() { date +%s; }
report() { echo "[$1] done in $(( $(stamp) - $2 )) s"; }

# Compare regenerated hierarchy totals (run_hierarchy.py JSON files) with
# results/hierarchy_6x6.json; exit non-zero if any |difference| exceeds TOL.
compare_hierarchy() {
    "$PYTHON" - "$@" <<'PY'
import json, math, sys
tol = float(sys.argv[1])
cutoffs = [int(c) for c in sys.argv[2].split(",")]
stored = json.load(open("results/hierarchy_6x6.json"))["records"]
ref = {(r["boundary"], r["rank"], r["cutoff"]): r["A"] for r in stored}
worst, n = 0.0, 0
seen = set()
for path in sys.argv[3:]:
    res = json.load(open(path))
    expected = {f"r{r}c{c}" for r in (4, 5, 6) for c in cutoffs}
    if res.get("status") != "complete" or res.get("geometry") != [6, 6] or set(res.get("totals") or {}) != expected:
        raise SystemExit(f"Incomplete or unexpected hierarchy output: {path}")
    boundary = res["boundary_label"]
    if boundary in seen:
        raise SystemExit(f"Duplicate boundary output: {boundary}")
    seen.add(boundary)
    for key, (re, im) in res["totals"].items():
        if not (math.isfinite(re) and math.isfinite(im)) or abs(im) > 1e-6:
            raise SystemExit(f"Nonfinite or unexpectedly complex total: {path} {key}")
        r, c = (int(x) for x in key[1:].split("c"))
        d = re - ref[(res["boundary_label"], r, c)]
        worst, n = max(worst, abs(d)), n + 1
        print(f"  6x6 {res['boundary_label']} {key}: A = {re:.12f}  stored diff {d:+.2e}")
ok = n > 0 and worst <= tol
print(f"  {n} totals compared, max |diff| = {worst:.2e} (tolerance {tol:g}): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
}

case "$TIER" in
verify)
    t0=$(stamp)
    "$PYTHON" -m pytest -q -p no:cacheprovider tests
    "$PYTHON" verify.py
    "$PYTHON" scripts/estimate_error.py --check
    report verify "$t0"
    ;;
cheap)
    need_gpu
    OUT="${OUT:-runs/cheap}"
    mkdir -p "$OUT"
    t0=$(stamp)
    # Exact small lattices, all four boundary sectors; each run exits non-zero if any
    # sector differs from the stored reference by more than 1e-10.
    "$PYTHON" scripts/run_exact.py --lattice 4x2 --backend cpu --out "$OUT/exact_4x2.json"
    "$PYTHON" scripts/run_exact.py --lattice 4x4 --backend gpu --out "$OUT/exact_4x4.json"
    # One 6x6 boundary sector of the hierarchy at support cutoff 4 (ranks 4, 5, 6),
    # in the production arithmetic, compared with the stored totals.
    "$PYTHON" scripts/run_hierarchy.py --geometry 6x6 --boundary PP --rank 4,5,6 --cutoff 4 \
        --device gpu --out "$OUT/hierarchy_6x6_PP_c4.json"
    compare_hierarchy 1e-8 4 "$OUT/hierarchy_6x6_PP_c4.json"
    report cheap "$t0"
    ;;
full)
    need_gpu
    OUT="${OUT:-runs/full}"
    CACHE_DIR="${CACHE_DIR:-cache}"
    mkdir -p "$OUT" "$CACHE_DIR"
    t0=$(stamp)
    # All four boundary sectors, ranks 4-6, cutoffs 4-6; six-site sources are cached
    # in CACHE_DIR and reused across sectors.
    for b in PP AP PA AA; do
        "$PYTHON" scripts/run_hierarchy.py --geometry 6x6 --boundary "$b" --rank 4,5,6 --cutoff 4,5,6 \
            --device gpu --cache-dir "$CACHE_DIR" --out "$OUT/hierarchy_6x6_$b.json"
    done
    compare_hierarchy 1e-8 4,5,6 "$OUT"/hierarchy_6x6_{PP,AP,PA,AA}.json
    report full "$t0"
    ;;
*)
    usage
    ;;
esac
