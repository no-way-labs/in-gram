"""Phase 30 analysis, as pre-registered in docs/PREREG_PHASE30.md: conf29 (locker, 56.0M) vs conf30-hyb54
(3:1 hybrid, 12 x d384, 54.2M), seeds 20-27, paired. D3 primary, D3c (both + free n-gram cache) secondary,
one-sided Wilcoxon, Holm across the two, 95% bootstrap CI over seeds.

    python scripts/analyze_phase30.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_phase29 import boot_ci, holm, scores  # noqa: E402

if __name__ == "__main__":
    z = np.load("runs/tags/cache16k.npz")
    B, Q = z["bucket"], z["q"]
    rows = {sd: (scores(f"conf29-o12481632-s{sd}", B, Q), scores(f"conf30-hyb54-s{sd}", B, Q)) for sd in range(20, 28)}
    print("| seed | locker 56M | hybrid 54M | locker + cache | hybrid + cache |")
    print("|---|---|---|---|---|")
    for sd, (l, h) in rows.items():
        print(f"| {sd} | {l[0]:.4f} | {h[0]:.4f} | {l[1]:.4f} | {h[1]:.4f} |")
    tests = {"D3 locker - hybrid54 (primary)": np.array([l[0] - h[0] for l, h in rows.values()]),
             "D3c both + cache": np.array([l[1] - h[1] for l, h in rows.values()])}
    ps = np.array([wilcoxon(d, alternative="less").pvalue for d in tests.values()])
    for (name, d), p, pa in zip(tests.items(), ps, holm(ps)):
        lo, hi = boot_ci(d)
        print(f"{name}: mean {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], {int((d < 0).sum())}/8 negative, "
              f"Wilcoxon p = {p:.4f}, Holm p = {pa:.4f}")
