"""N-gram repeat statistics of held-out agentic trajectories as a function of position in the session
(data only, no model; CPU, vectorized).

Question: In-gram reads what followed earlier occurrences of the current N-gram. Are the statistics it
depends on the same deeper into a session as in the first 16k tokens? For held-out SWE-rebench trajectories,
read from their first token and cut at 262,144, we report per position bucket (position = index of the
predicted token in the session):
  seen_N      share of tokens whose preceding N-gram occurred earlier in the session (N = In-gram's cue lengths)
  hit_N       share where the continuation of the most recent earlier occurrence is the true next token
  lookup8     the paper's lookup: longest earlier match with N = 8 down to 1, most recent continuation
  lookup32    the same over In-gram's cue lengths, 32 down to 1
  longest     distribution of the longest matched cue length (none, 1, 2, 4, 8, 16, 32)
  far         for N = 8 matches: share whose most recent occurrence lies more than 512 / 16,384 tokens back
  kinds (N=4) consistent (one earlier continuation, right), latest (several, the most recent is right),
              older (several, an older one is right), new continuation (cue seen, continuation never),
              unseen cue
  new slots   cues written for the first time per token, summed over the six cue lengths (marginal slot rate)

    .venv/bin/python scripts/gram_stats_by_context.py [n_trajectories=300] [min_length=0]
min_length keeps only trajectories at least that long, so that every position bucket up to it pools the same
trajectories (separates position in the session from which trajectories are long).
Writes results/gram_stats_by_context.json (or ..._min<min_length>.json).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import agentic  # noqa: E402

CUES = (1, 2, 4, 8, 16, 32)
NS = tuple(range(1, 9)) + (16, 32)
CAP = 262144
EDGES = np.array([0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144])
MULT = np.random.default_rng(12345).integers(1, 2**63, size=32, dtype=np.uint64) | np.uint64(1)
PAIR = np.uint64(0x9E3779B97F4A7C15)


def prev_same(h):
    """Index of the previous position with the same value, or -1 (h is 1-D)."""
    order = np.argsort(h, kind="stable")
    hs = h[order]
    same = hs[1:] == hs[:-1]
    prev = np.full(len(h), -1, dtype=np.int64)
    prev[order[1:]] = np.where(same, order[:-1], -1)
    return prev, order, np.concatenate([[False], same])


def stats(w):
    """Per predicted position p = t + 1 (t = end of the cue): dict of arrays of length L (unused entries False/0)."""
    L = len(w)
    w64 = w.astype(np.uint64)
    out = {}
    for N in NS:
        seen = np.zeros(L, bool)
        hit = np.zeros(L, bool)
        dist = np.zeros(L, np.int64)
        if L > N:
            h = np.lib.stride_tricks.sliding_window_view(w64, N)[: L - N] @ MULT[:N]  # cue ending at t = i + N - 1
            prev, order, same_sorted = prev_same(h)
            i = np.arange(L - N)
            p = i + N  # predicted position
            ok = prev >= 0
            seen[p] = ok
            hit[p] = ok & (w[np.where(ok, prev, 0) + N] == w[p])
            dist[p] = np.where(ok, i - prev, 0)
            if N == 4:  # kinds of continuation
                pair = h * PAIR + w64[p]
                pprev, _, _ = prev_same(pair)
                first = (pprev < 0).astype(np.int64)  # this (cue, continuation) pair is new
                fs = first[order]
                cs = np.cumsum(fs) - fs
                base = np.maximum.accumulate(np.where(~same_sorted, cs, 0))
                ncont = np.zeros(L - N, np.int64)
                ncont[order] = cs - base  # distinct continuations of this cue before i
                kinds = np.zeros(L, np.int8)  # 0 unseen cue, 1 consistent, 2 latest, 3 older, 4 new continuation
                k = np.where(~ok, 0, np.where(pprev < 0, 4, np.where(hit[p], np.where(ncont == 1, 1, 2), 3)))
                kinds[p] = k
                out["kinds"] = kinds
        out[f"seen{N}"], out[f"hit{N}"] = seen, hit
        if N == 8:
            out["dist8"] = dist
    return out


def main(n_traj, min_len=0):
    t0 = time.time()
    corpus = agentic.load("validation", "gpt2", max_rows=16384)
    lens = np.asarray(corpus.lengths)
    q = np.percentile(lens, [10, 25, 50, 75, 90, 99])
    print(f"held-out trajectories: {len(lens)}, tokens {lens.sum() / 1e6:.1f}M; length percentiles 10/25/50/75/90/99: "
          + " / ".join(f"{x / 1000:.0f}k" for x in q) + f"; max {lens.max() / 1000:.0f}k")
    print("share of trajectories at least: " + ", ".join(f"{e // 1024}k: {np.mean(lens >= e):.1%}" for e in EDGES[5:]))
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(lens >= min_len)
    pick = rng.choice(pool, size=min(n_traj, len(pool)), replace=False)
    B = len(EDGES) - 1
    acc = {k: np.zeros(B) for k in ["n", "traj", "lookup8", "lookup32", "far512", "far16k", "seen8n", "newslots"]
           + [f"seen{N}" for N in CUES] + [f"hit{N}" for N in CUES] + [f"long{N}" for N in (0,) + CUES]
           + [f"kind{j}" for j in range(5)]}
    for j, ti in enumerate(pick):
        w = np.asarray(corpus.get(int(ti)))[:CAP].astype(np.int64)
        s = stats(w)
        L = len(w)
        pos = np.arange(L)
        b = np.searchsorted(EDGES, pos, side="right") - 1
        valid = pos >= 1
        # longest-match lookups
        def backoff(ns):
            decided = np.zeros(L, bool)
            ok = np.zeros(L, bool)
            longest = np.zeros(L, np.int64)
            for N in sorted(ns, reverse=True):
                take = s[f"seen{N}"] & ~decided
                ok |= take & s[f"hit{N}"]
                longest[take] = N
                decided |= take
            return ok, longest
        ok8, _ = backoff(range(1, 9))
        ok32, longest = backoff(CUES)
        cols = {"n": valid, "lookup8": ok8, "lookup32": ok32,
                "far512": s["seen8"] & (s["dist8"] > 512), "far16k": s["seen8"] & (s["dist8"] > 16384), "seen8n": s["seen8"],
                "newslots": sum((~s[f"seen{N}"] & (pos >= N)).astype(np.int64) for N in CUES),
                **{f"seen{N}": s[f"seen{N}"] for N in CUES}, **{f"hit{N}": s[f"hit{N}"] for N in CUES},
                **{f"long{N}": valid & (longest == N) for N in (0,) + CUES},
                **{f"kind{k}": (pos >= 4) & (s["kinds"] == k) for k in range(5)}}
        for k, v in cols.items():
            acc[k] += np.bincount(b, weights=v.astype(np.float64), minlength=B)[:B]
        acc["traj"] += np.bincount(np.unique(b), minlength=B)[:B]
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(pick)} trajectories, {time.time() - t0:.0f} s", flush=True)
    n = np.maximum(acc["n"], 1)
    names = [f"{EDGES[i] // 1024}k-{EDGES[i + 1] // 1024}k" for i in range(B)]
    rows = {"bucket": names, "trajectories": acc["traj"].astype(int).tolist(), "tokens": acc["n"].astype(int).tolist()}
    rows["corpus"] = dict(trajectories=int(len(lens)), tokens=int(lens.sum()), median=float(np.median(lens)),
                          max=int(lens.max()), share_ge_64k=float(np.mean(lens >= 65536)),
                          share_ge_128k=float(np.mean(lens >= 131072)), used=int(len(pick)), min_len=int(min_len))
    rows["upper"] = EDGES[1:].tolist()
    for k in acc:
        if k in ("n", "traj"):
            continue
        rows[k] = (acc[k] / n).tolist()
    for N in CUES:
        rows[f"precision{N}"] = (acc[f"hit{N}"] / np.maximum(acc[f"seen{N}"], 1)).tolist()
    rows["far512_of_matches"] = (acc["far512"] / np.maximum(acc["seen8n"], 1)).tolist()
    rows["far16k_of_matches"] = (acc["far16k"] / np.maximum(acc["seen8n"], 1)).tolist()

    def table(title, keys, fmt="{:.3f}"):
        print(f"\n### {title}")
        print("| position | " + " | ".join(k for k, _ in keys) + " |")
        print("|---|" + "---|" * len(keys))
        for i in range(B):
            if acc["n"][i] == 0:
                continue
            print(f"| {names[i]} | " + " | ".join(fmt.format(rows[c][i]) for _, c in keys) + " |")

    print(f"\n{len(pick)} trajectories of at least {min_len} tokens, read from the first token, cut at {CAP}")
    print("| position | trajectories reaching it | tokens |\n|---|---|---|")
    for i in range(B):
        print(f"| {names[i]} | {rows['trajectories'][i]} | {rows['tokens'][i]} |")
    table("Cue seen earlier in the session (share of tokens), by cue length N", [(f"N={N}", f"seen{N}") for N in CUES])
    table("Most recent continuation is right, given the cue was seen (precision), by N", [(f"N={N}", f"precision{N}") for N in CUES])
    table("Lookup accuracy (share of all tokens)", [("N = 8..1 (paper)", "lookup8"), ("cue lengths 32..1", "lookup32")])
    table("Longest matched cue length (share of tokens)", [("none", "long0")] + [(f"{N}", f"long{N}") for N in CUES])
    table("N = 8 matches by distance to the most recent occurrence", [("share of tokens with a match", "seen8n"),
          ("of matches: more than 512 back", "far512_of_matches"), ("of matches: more than 16,384 back", "far16k_of_matches")])
    table("Kinds of continuation at N = 4 (share of tokens)", [("unseen cue", "kind0"), ("consistent", "kind1"),
          ("latest value", "kind2"), ("older value", "kind3"), ("new continuation", "kind4")])
    table("New slots per token (six cue lengths; bound 6)", [("new slots", "newslots")], "{:.2f}")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/gram_stats_by_context{'_min' + str(min_len) if min_len else ''}.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
