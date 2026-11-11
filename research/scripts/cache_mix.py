"""Model + free in-context N-gram cache (a cache LM) on the existing per-token NLL files.

For each scored position t of the 32 x 16k eval windows, the cache finds the longest N in 8..1 whose
N-gram ending at t-1 occurred earlier in the window and predicts from the counts of its earlier
continuations: q(true) = count(true) / count(all). The mixture's probability of the true token is
(1 - lam_N) p_model(true) + lam_N q(true), so p_model(true) = exp(-nll) is all we need.
lam_N (one per matched N, 0 when nothing matched) is fit on one half of the windows and scored on the
other, both ways (2-fold cross-fit), so the reported mixed NLL never sees its own tuning windows.

    python scripts/cache_mix.py runs/sd14-a2p0-s0 runs/sd14-a0p1-s0 ...
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tag_eval_set import eval_windows  # noqa: E402

NMAX = 8
LAMS = np.linspace(0, 0.95, 39)


def cache_probs(w):
    """bucket[t] in 0..NMAX (0 = no match), q[t] = cache probability of w[t]."""
    L = len(w)
    tabs = [defaultdict(Counter) for _ in range(NMAX + 1)]
    bucket = np.zeros(L, np.int8)
    q = np.zeros(L, np.float32)
    for t in range(1, L):
        for N in range(min(NMAX, t - 1), 0, -1):  # context w[t-N..t-1], all scored tokens (index >= 1)
            c = tabs[N].get(tuple(w[t - N:t]))
            if c:
                bucket[t] = N
                q[t] = c[w[t]] / sum(c.values())
                break
        for N in range(1, min(NMAX, t - 1) + 1):
            tabs[N][tuple(w[t - N:t])][w[t]] += 1
    return bucket, q


def fit(p, q, b, idx):
    lam = np.zeros(NMAX + 1)
    for N in range(1, NMAX + 1):
        m = b[idx] == N
        if m.any():
            pp, qq = p[idx][m][:, None], q[idx][m][:, None]
            lam[N] = LAMS[np.argmin(-np.log((1 - LAMS) * pp + LAMS * qq + 1e-12).mean(0))]
    return lam


if __name__ == "__main__":
    cache = Path("runs/tags/cache16k.npz")
    if cache.exists():
        z = np.load(cache)
        B, Q = z["bucket"], z["q"]
    else:
        task = dict(name="agentic", L_max=16384, tokenizer="gpt2", max_rows=16384)
        wins = [np.where(w < 0, -1, w).tolist() for w in eval_windows(task, 32, 1)]
        out = [cache_probs(w) for w in wins]
        B, Q = np.stack([a for a, _ in out]), np.stack([b for _, b in out])
        np.savez_compressed(cache, bucket=B, q=Q)
    print(f"cache: matched {(B[:, 1:] > 0).mean():.3f} of tokens, mean q(true | matched) {Q[B > 0].mean():.3f}")
    print("| model | NLL | + cache (2-fold) | change | lam_8 / lam_4 / lam_1 |")
    print("|---|---|---|---|---|")
    for d in sys.argv[1:]:
        f = next(Path(d).glob("*.npz"))
        nll = np.load(f)["nll"].astype(np.float64)
        ok = ~np.isnan(nll)
        halves = [np.arange(16), np.arange(16, 32)]
        mixed = np.zeros_like(nll)
        for tr, te in [(halves[0], halves[1]), (halves[1], halves[0])]:
            okt = ok[tr]
            p = np.exp(-nll[tr][okt]); q = Q[tr][okt]; b = B[tr][okt]
            lam = fit(p, q, b, slice(None))
            lt = lam[B[te]]
            mixed[te] = -np.log((1 - lt) * np.exp(-nll[te]) + lt * Q[te] + 1e-12)
        base, mix = nll[ok].mean(), mixed[ok].mean()
        lam_all = fit(np.exp(-nll[ok]), Q[ok], B[ok], slice(None))
        print(f"| {Path(d).name} | {base:.4f} | {mix:.4f} | {mix - base:+.4f} | "
              f"{lam_all[8]:.2f} / {lam_all[4]:.2f} / {lam_all[1]:.2f} |")
