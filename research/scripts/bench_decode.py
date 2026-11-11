"""Decode benchmark (src/bench_decode.py) from the command line.

  python scripts/bench_decode.py                       # defaults: 4k..1M, 256 tokens, 5 arms
  python scripts/bench_decode.py --lengths 4096 16384 --arms attn gdn --out runs/bench.json

On Rhombic it runs as the "bench-decode" campaign (gpu_job, kind "bench_decode").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bench_decode import DEFAULT_ARMS, bench  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+")
    ap.add_argument("--arms", nargs="+", choices=sorted(DEFAULT_ARMS))
    ap.add_argument("--n-decode", type=int)
    ap.add_argument("--tokens", choices=["wikitext", "random"])
    ap.add_argument("--device")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = {}
    if a.lengths:
        cfg["lengths"] = a.lengths
    if a.arms:
        cfg["arms"] = {k: DEFAULT_ARMS[k] for k in a.arms}
    if a.n_decode:
        cfg["n_decode"] = a.n_decode
    if a.tokens:
        cfg["tokens"] = a.tokens
    if a.device:
        cfg["device"] = a.device
    res = bench(cfg)
    text = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
