"""Where the locker-vs-hybrid gap sits, by token type, at 10k and 20k training steps (exploratory, post hoc).
Pairs (seeds 20-27): 10k = conf29 vs conf6-a2p0; 20k = b31-longloc vs b31-longhyb. Per type: mean NLL of each
arm (seed-averaged per token), gap, share of the overall gap, 95% bootstrap CI over the 32 windows.
    python scripts/gap_by_type.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_phase29 import nll_file  # noqa: E402
from src.tagger import TAGS  # noqa: E402


def arm(prefix):
    return np.mean([np.load(nll_file(f"{prefix}-s{s}"))["nll"].astype(np.float64) for s in range(20, 28)], 0)


tags = np.load("runs/tags/eval16k.npz")
tags = {"all": np.ones_like(tags[TAGS[0]], bool), **{k: tags[k] for k in TAGS}}
pairs = {"10k": ("conf29-o12481632", "conf6-a2p0"), "20k": ("b31-longloc", "b31-longhyb")}
A = {k: (arm(a), arm(b)) for k, (a, b) in pairs.items()}
rng = np.random.default_rng(0)
idx = rng.integers(0, 32, (10000, 32))
ok = ~np.isnan(A["10k"][0])
N = ok.sum()
print("| type | share | hybrid 10k | locker 10k | gap 10k | hybrid 20k | locker 20k | gap 20k [95% CI] | contrib. 20k | gap kept 20k/10k |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k, m in tags.items():
    m = m & ok
    row = [k, f"{m.mean():.1%}"]
    g = {}
    for t in ("10k", "20k"):
        loc, hyb = A[t]
        g[t] = loc[m].mean() - hyb[m].mean()
        row += [f"{hyb[m].mean():.3f}", f"{loc[m].mean():.3f}"]
        if t == "10k":
            row.append(f"{g[t]:+.3f}")
    loc, hyb = A["20k"]
    d = np.where(m, loc - hyb, 0)
    s, c = d.sum(1), m.sum(1)
    bs = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    row[-2:] = row[-2:]  # keep hybrid/locker 20k
    row.append(f"{g['20k']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    row.append(f"{d.sum() / N:+.4f}")
    row.append(f"{g['20k'] / g['10k']:.2f}" if abs(g["10k"]) > 1e-4 else "-")
    print("| " + " | ".join(row) + " |")
