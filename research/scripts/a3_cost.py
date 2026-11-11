"""PREREG_PHASE27 A3: per-type cost of removing attention, C(type, P) = NLL(A=0, P) - NLL(A=2, P),
seeds averaged per token, 95% bootstrap CI over the 32 eval windows (10k resamples, windows resampled
jointly across arms). Also prints per-type NLL per cell. Reads runs/sd14-a{A}p{P}-s{0,1}/*.npz.

    python scripts/a3_cost.py [runs/tags/eval16k.npz]
"""
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tagger import TAGS  # noqa: E402


GRID = __import__("os").environ.get("GRID", "sd14")  # overlay cells (P >= 1) from this grid; P = 0 always sd14


def cell(A, P):
    prefix = GRID if P else "sd14"
    fs = sorted(glob.glob(f"runs/{prefix}-a{A}p{P}-s[01]/*.npz"))
    assert len(fs) == 2, (A, P, fs)
    return np.mean([np.load(f)["nll"].astype(np.float64) for f in fs], axis=0)  # (windows, T)


def per_window(nll, mask):
    """Per-window sums and counts, so a resample's mean is sum(sums) / sum(counts)."""
    m = mask & ~np.isnan(nll)
    return np.where(m, nll, 0).sum(1), m.sum(1)


def main(tag_path="runs/tags/eval16k.npz", B=10_000, seed=0):
    tags = np.load(tag_path)
    tags = {"all": np.ones_like(tags[TAGS[0]], bool), **{k: tags[k] for k in TAGS}}
    rng = np.random.default_rng(seed)
    W = tags["all"].shape[0]
    idx = rng.integers(0, W, size=(B, W))
    cells = {(A, P): cell(A, P) for A in (0, 2) for P in (0, 1, 2, 4)}
    print("| type | share | " + " | ".join(f"C(P={P}) [95% CI]" for P in (0, 1, 2, 4)) + " |")
    print("|---|---|" + "---|" * 4)
    for k, m in tags.items():
        row = [k, f"{m.mean():.1%}"]
        for P in (0, 1, 2, 4):
            s0, c0 = per_window(cells[0, P], m)
            s2, c2 = per_window(cells[2, P], m)
            point = s0.sum() / c0.sum() - s2.sum() / c2.sum()
            bs = s0[idx].sum(1) / c0[idx].sum(1) - s2[idx].sum(1) / c2[idx].sum(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row.append(f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print("| " + " | ".join(row) + " |")
    print()
    print("Per-type NLL, seed means:")
    print("| cell | " + " | ".join(tags) + " |")
    print("|---|" + "---|" * len(tags))
    for (A, P), nll in sorted(cells.items()):
        print(f"| A{A}P{P} | " + " | ".join(f"{np.nanmean(nll[m]):.3f}" for m in tags.values()) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:2])
