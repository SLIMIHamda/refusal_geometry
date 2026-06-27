"""Populate data/<name>.jsonl from the HF Hub. Host-run (needs `datasets` + network).

    python scripts/download_benchmarks.py --benchmarks harmbench xstest --limit 200
    python scripts/download_benchmarks.py --all
"""
from __future__ import annotations

import argparse
import sys

from asw.data.download import SPECS, download_benchmark


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="download benchmarks as harness JSONL")
    p.add_argument("--benchmarks", nargs="*", default=[], choices=sorted(SPECS))
    p.add_argument("--all", action="store_true")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    names = sorted(SPECS) if args.all else args.benchmarks
    if not names:
        p.error("pass --benchmarks <names...> or --all")
    for name in names:
        try:
            path = download_benchmark(name, out_dir=args.out_dir, limit=args.limit)
            print(f"[ok] {name:14s} -> {path}")
        except Exception as e:  # noqa: BLE001 - report and continue with the rest
            print(f"[fail] {name:14s}: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
