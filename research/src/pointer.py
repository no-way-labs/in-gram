"""Pointer-memory ("locker3") statistics on a token stream: longest earlier match, copy run lengths,
and confidence vs match length. CPU only, numpy only, so remote jobs can import it.

For each position t (predicting tok[t]) we find the longest suffix of tok[:t], over a ladder of
lengths, that occurred earlier, using rolling hashes; each length keeps its most recent earlier
occurrence. The pointer predicts the token that followed that occurrence.

`match_lengths` is the per-position core (used for loss breakdowns by pointer regime, see
src/agentic.py); `analyze` adds the aggregate statistics used by scripts/pointer_stats.py.
"""
from __future__ import annotations

import numpy as np

LADDER = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
P = np.uint64(1_000_003)
SHORT_MAX = 7  # regime "short" = match length 1..7, "long" = 8+
REGIME_NONE, REGIME_SHORT, REGIME_LONG = 0, 1, 2
REGIME_NAMES = ("none", "short", "long")


def suffix_keys(tok: np.ndarray, ladder=LADDER) -> dict[int, np.ndarray]:
    """keys[L][t] = hash of tok[t-L:t] (suffix ending at t-1), uint64 wraparound; 0 if t < L.

    Polynomial hash sum_j x[t-L+j] * P^(L-1-j) mod 2^64 with x = tok + 1, computed directly per
    length (vectorized; equal to the prefix-hash difference H[t] - H[t-L] * P^L mod 2^64)."""
    T = len(tok)
    x = np.asarray(tok).astype(np.uint64) + np.uint64(1)
    out = {}
    with np.errstate(over="ignore"):
        for L in ladder:
            k = np.zeros(T + 1, dtype=np.uint64)
            if T >= L:
                acc = np.zeros(T - L + 1, dtype=np.uint64)
                for j in range(L):  # Horner over the L tokens of each suffix
                    acc = acc * P + x[j : T - L + 1 + j]
                k[L:] = acc
            out[L] = k ^ np.uint64(L * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF)
    return out


def _prev_equal(k: np.ndarray) -> np.ndarray:
    """prev[i] = largest i' < i with k[i'] == k[i], else -1."""
    order = np.argsort(k, kind="stable")  # equal keys stay in index order
    ks = k[order]
    same = np.zeros(len(k), dtype=bool)
    same[1:] = ks[1:] == ks[:-1]
    prev = np.full(len(k), -1, dtype=np.int64)
    prev[order[same]] = order[np.flatnonzero(same) - 1]
    return prev


def match_lengths(tok: np.ndarray, ladder=LADDER) -> tuple[np.ndarray, np.ndarray]:
    """(match_len, src), both int64 [T].

    match_len[t] = longest ladder length L such that tok[t-L:t] also ended at some earlier position
    (0 if none); src[t] = the position whose token the pointer predicts (the token after the most
    recent earlier occurrence of that suffix), -1 if none. Position 0 never matches."""
    tok = np.asarray(tok, dtype=np.int64)
    T = len(tok)
    keys = suffix_keys(tok, ladder)
    match_len = np.zeros(T, dtype=np.int64)
    src = np.full(T, -1, dtype=np.int64)
    for L in ladder:  # ascending, so the longest match wins
        if T <= L:
            break
        prev = _prev_equal(keys[L][L:T])  # keys for t in [L, T)
        hit = prev >= 0
        t = np.flatnonzero(hit) + L
        match_len[t] = L
        src[t] = prev[hit] + L
    return match_len, src


def regime(match_len: np.ndarray) -> np.ndarray:
    """int8 regime per position: 0 none, 1 short (1..7), 2 long (8+)."""
    out = np.zeros(np.shape(match_len), dtype=np.int8)
    out[(match_len >= 1) & (match_len <= SHORT_MAX)] = REGIME_SHORT
    out[match_len > SHORT_MAX] = REGIME_LONG
    return out


def analyze(tok: np.ndarray, min_lens=(4, 8), window=262_144) -> dict:
    tok = np.asarray(tok, dtype=np.int64)
    T = len(tok)
    match_len, src = match_lengths(tok)
    has = src >= 0
    right = np.zeros(T, dtype=bool)
    right[has] = tok[src[has]] == tok[has]
    out = dict(tokens=int(T), matched=float(has.mean()))
    out["match_len_dist"] = {str(L): float((match_len == L).mean()) for L in LADDER}
    out["p_right_by_len"] = {str(L): float(right[match_len == L].mean()) if (match_len == L).any() else None
                             for L in LADDER}
    out["p_right_len_ge"] = {str(m): float(right[match_len >= m].mean()) for m in (8, 16, 32, 64)
                             if (match_len >= m).any()}
    d = np.arange(T) - src
    out["source_beyond_window"] = float((has & (d > window)).mean())
    out["source_beyond_window_right"] = float((has & (d > window) & right).mean())
    out["source_ge_4k"] = float((has & (d >= 4096)).mean())
    for m in min_lens:  # cursor simulation
        runs, covered, cur, run = [], 0, -1, 0
        for t in range(1, T):
            if cur < 0 and match_len[t] >= m:
                cur, run = int(src[t]), 0
            if cur >= 0:
                if cur < t and tok[cur] == tok[t]:
                    covered += 1
                    run += 1
                    cur += 1
                else:
                    runs.append(run)
                    cur = -1
        if cur >= 0:
            runs.append(run)
        r = np.asarray(runs) if runs else np.zeros(1)
        out[f"cursor_min{m}"] = dict(anchors=len(runs), coverage=covered / T, mean_run=float(r.mean()),
                                     median_run=float(np.median(r)), frac_runs_ge8=float((r >= 8).mean()),
                                     tokens_in_runs_ge8=float(r[r >= 8].sum() / T),
                                     tokens_in_runs_ge32=float(r[r >= 32].sum() / T))
    return out
