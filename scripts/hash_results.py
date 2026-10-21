#!/usr/bin/env python3
"""Write (or check) results/SHA256SUMS for every file under results/.

    python scripts/hash_results.py [--results DIR] [--check]

The format is that of ``sha256sum`` (``<hex>  <relative path>``), so the file can
also be checked with ``cd results && sha256sum -c SHA256SUMS``.
"""
import argparse
import hashlib
import sys
from pathlib import Path

NAME = "SHA256SUMS"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def listing(results: Path) -> str:
    files = sorted(p for p in results.rglob("*") if p.is_file() and p.name != NAME
                   and "__pycache__" not in p.parts)
    return "".join(f"{digest(p)}  {p.relative_to(results).as_posix()}\n" for p in files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = listing(args.results)
    target = args.results / NAME
    if args.check:
        ok = target.exists() and target.read_text(encoding="utf-8") == text
        print("SHA256SUMS ok" if ok else "SHA256SUMS MISMATCH")
        return 0 if ok else 1
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target} ({text.count(chr(10))} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
