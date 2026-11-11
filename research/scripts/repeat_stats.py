"""Repeat structure of the 16k agentic eval windows (same windows as post_eval / tag_eval_set.py).

For each scored position t and N in 1..8: does the N-gram ending at t occur earlier in the window
("seen"), and does its most recent earlier continuation equal the true next token ("lookup_hit",
what an N-gram prompt-lookup draft would get right)? Also the longest-match (backoff 8..1) hit rate.
Output: one line per N with the token-weighted share and the 32-window spread (min / median / max)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tag_eval_set import eval_windows  # noqa: E402


def lookup(w, N):
    """seen[t], hit[t] for predicting w[t+1] from the N-gram w[t-N+1..t]."""
    L = len(w)
    seen = np.zeros(L, bool)
    hit = np.zeros(L, bool)
    last = {}
    for t in range(N - 1, L - 1):
        k = tuple(w[t - N + 1:t + 1])
        if k in last:
            seen[t] = True
            hit[t] = w[last[k] + 1] == w[t + 1]
        last[k] = t
    return seen, hit


if __name__ == "__main__":
    task = dict(name="agentic", L_max=16384, tokenizer="gpt2", max_rows=16384)
    wins = [np.where(w < 0, 0, w) for w in eval_windows(task, 32, 1)]
    Ns = range(1, 9)
    res = {N: [lookup(w, N) for w in wins] for N in Ns}
    print("N  seen   hit    hit|seen  (per-window hit: min / med / max)")
    for N in Ns:
        s = np.concatenate([a for a, _ in res[N]]).mean()
        h = np.array([b.mean() for _, b in res[N]])
        hh = np.concatenate([b for _, b in res[N]]).mean()
        print(f"{N}  {s:.3f}  {hh:.3f}  {hh / s:.3f}     {h.min():.3f} / {np.median(h):.3f} / {h.max():.3f}")
    # longest match, backoff from 8 to 1: predict with the longest N whose N-gram was seen
    bo = []
    for i in range(len(wins)):
        pred_hit = np.zeros(len(wins[i]), bool)
        decided = np.zeros(len(wins[i]), bool)
        for N in reversed(Ns):
            s, h = res[N][i]
            take = s & ~decided
            pred_hit |= take & h
            decided |= take
        bo.append((decided.mean(), pred_hit.mean()))
    bo = np.array(bo)
    print(f"backoff 8..1: drafted {bo[:, 0].mean():.3f}, correct {bo[:, 1].mean():.3f} "
          f"(per-window correct {bo[:, 1].min():.3f} / {np.median(bo[:, 1]):.3f} / {bo[:, 1].max():.3f})")
