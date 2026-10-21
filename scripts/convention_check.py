#!/usr/bin/env python3
"""Reproduce the experimental two-step value of A from the published raw shots.

Downloads ``shots_and_analysis.zip`` of the authors' data record
(Zenodo, DOI 10.5281/zenodo.17483197, https://zenodo.org/records/17483197)
into a cache directory, verifies its MD5 checksum against the record
metadata, and recomputes the raw-shot estimator directly from the shot JSON
inside the archive (nothing from the archive is executed or extracted).

Estimator (per shot; 36 sites, species-major labels: up site i is ``s_i``,
down site i is ``s_{i+36}``; ``leaked_s_q`` marks a leaked qubit).  A site is
usable when both of its bits are present; ``d_i = b_up - b_down``.  For the
``K`` usable sites

    a_shot = 35 / (K (K-1)) * [ sum_i d_i^2 - (sum_i d_i)^2 ]
           = -35 * mean over unordered usable pairs i<j of (d_i d_j),

each shot has equal weight and the uncertainty is the sample standard error
``std(a, ddof=1) / sqrt(shots)``.  In the fixed-number sector this estimates
``A = (2/N) [<D> - <eta^dag eta>]`` (positive convention of this package).
Expected: A = 0.2658404483 +/- 0.0254870558 (400 shots).

Also printed: the paper's convention ``P = -A`` and the value of the
expression written in the tracker issue, ``-A/2``.

Usage::

    python scripts/convention_check.py [--cache data/zenodo] [--all-steps]
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys
import urllib.request
import zipfile
from fractions import Fraction

RECORD = "17483197"
API = f"https://zenodo.org/api/records/{RECORD}"
ARCHIVE = "shots_and_analysis.zip"
SHOTS = "single_layer_half_filled/shots/"
PATTERNS = {2: "etadiag1-eta", 3: "etadiag-extra1step-eta", 4: "etadiag-extra2step-eta"}
EXPECTED = {2: (400, 0.26584044825669506, 0.025487055830787846)}


def fetch(cache: pathlib.Path) -> pathlib.Path:
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "record.json"
    if not meta_path.exists():
        with urllib.request.urlopen(API, timeout=60) as r:
            meta_path.write_bytes(r.read())
    meta = json.loads(meta_path.read_text())
    entry = next(f for f in meta["files"] if f["key"] == ARCHIVE)
    path = cache / ARCHIVE
    if not path.exists() or path.stat().st_size != entry["size"]:
        print(f"downloading {entry['links']['self']}", flush=True)
        with urllib.request.urlopen(entry["links"]["self"], timeout=300) as r:
            path.write_bytes(r.read())
    md5 = "md5:" + hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != entry["checksum"]:
        raise SystemExit(f"checksum mismatch for {path}: {md5} != {entry['checksum']}")
    return path


def shot_value(shot: dict) -> Fraction:
    bits = []
    for q in range(90):
        normal, leaked = f"s_{q}", f"leaked_s_{q}"
        if (normal in shot) + (leaked in shot) != 1:
            raise ValueError(f"qubit {q}: expected exactly one of {normal}, {leaked}")
        bits.append(shot[normal][0] if normal in shot else None)
    d = [bits[i] - bits[i + 36] for i in range(36)
         if bits[i] is not None and bits[i + 36] is not None]
    k = len(d)
    if k < 2:
        raise ValueError("shot with fewer than two usable sites")
    return Fraction(35 * (sum(x * x for x in d) - sum(d) ** 2), k * (k - 1))


def estimate(archive: pathlib.Path, steps: int):
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if n.startswith(SHOTS) and PATTERNS[steps] in n
                       and n.endswith(".json"))
        values = [shot_value(s) for n in names for s in json.loads(z.read(n))]
    m = len(values)
    mean = sum(values, Fraction()) / m
    var = sum((v - mean) ** 2 for v in values) / (m - 1)
    return dict(steps=steps, files=[n.rsplit("/", 1)[-1] for n in names], shots=m,
                A=float(mean), sem=math.sqrt(float(var) / m))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("data/zenodo"))
    p.add_argument("--all-steps", action="store_true", help="also evaluate the 3- and 4-step data")
    p.add_argument("--ideal", type=float, default=0.3836428102,
                   help="ideal 6x6 value of A to convert alongside (default: hierarchy estimate)")
    a = p.parse_args()
    archive = fetch(a.cache)
    ok = True
    for steps in ((2, 3, 4) if a.all_steps else (2,)):
        r = estimate(archive, steps)
        print(f"{steps} steps: {r['shots']} shots from {len(r['files'])} file(s)")
        print(f"  A        = {r['A']:.10f} +/- {r['sem']:.10f}   (positive convention)")
        print(f"  P = -A   = {-r['A']:.10f} +/- {r['sem']:.10f}   (paper convention)")
        print(f"  -A/2     = {-r['A'] / 2:.10f} +/- {r['sem'] / 2:.10f}   (expression in the issue)")
        if steps in EXPECTED:
            shots, mean, sem = EXPECTED[steps]
            match = r["shots"] == shots and abs(r["A"] - mean) < 1e-12 and abs(r["sem"] - sem) < 1e-12
            print(f"  matches published value {mean:.10f} +/- {sem:.10f}: {match}")
            ok = ok and match
    x = a.ideal
    print(f"ideal A = {x:.10f}  ->  P = -A = {-x:.10f},  -A/2 = {-x / 2:.10f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
