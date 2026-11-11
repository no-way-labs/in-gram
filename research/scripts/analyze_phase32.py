"""Phase 32 analysis as pre-registered in docs/PREREG_PHASE32.md (80k steps, seeds 20-27).
T1 = L7 - H3 (primary), T2 = L7 - H7; one-sided Wilcoxon (H1: L7 lower), Holm across the two; with-cache
versions, medians, 95% bootstrap CIs over seeds, and the per-type gaps for T1.
    python scripts/analyze_phase32.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_phase29 import boot_ci, holm, nll_file, scores  # noqa: E402
from src.tagger import TAGS  # noqa: E402

S = range(20, 28)
z = np.load("runs/tags/cache16k.npz")
B, Q = z["bucket"], z["q"]
sc = {c: [scores(f"p32-{c}-s{s}", B, Q) for s in S] for c in ("a1p1", "a2p0", "a1p0")}
print("| seed | 7:1 + locker | 3:1 | 7:1 |\n|---|---|---|---|")
for i, s in enumerate(S):
    print(f"| {s} | {sc['a1p1'][i][0]:.4f} | {sc['a2p0'][i][0]:.4f} | {sc['a1p0'][i][0]:.4f} |")
tests = {"T1 7:1+locker - 3:1 (primary)": "a2p0", "T2 7:1+locker - 7:1": "a1p0"}
ds = {n: np.array([a[0] - b[0] for a, b in zip(sc["a1p1"], sc[c])]) for n, c in tests.items()}
ps = np.array([wilcoxon(d, alternative="less").pvalue for d in ds.values()])
print()
for (n, d), p, pa in zip(ds.items(), ps, holm(ps)):
    lo, hi = boot_ci(d)
    print(f"{n}: mean {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], median {np.median(d):+.4f}, "
          f"{int((d < 0).sum())}/8 negative, Wilcoxon p = {p:.4f}, Holm p = {pa:.4f}")
for n, c in tests.items():
    d = np.array([a[1] - b[1] for a, b in zip(sc["a1p1"], sc[c])])
    lo, hi = boot_ci(d)
    print(f"{n} with cache (reported): {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], {int((d < 0).sum())}/8")
tags = np.load("runs/tags/eval16k.npz")
L7 = np.mean([np.load(nll_file(f"p32-a1p1-s{s}"))["nll"].astype(np.float64) for s in S], 0)
H3 = np.mean([np.load(nll_file(f"p32-a2p0-s{s}"))["nll"].astype(np.float64) for s in S], 0)
ok = ~np.isnan(L7)
print("\nT1 by type (seed-averaged per token):\n| type | share | 3:1 | 7:1 + locker | gap |\n|---|---|---|---|---|")
for k in ("all", *TAGS):
    m = ok if k == "all" else (tags[k] & ok)
    print(f"| {k} | {m.mean():.1%} | {H3[m].mean():.3f} | {L7[m].mean():.3f} | {L7[m].mean() - H3[m].mean():+.3f} |")
