"""Slots that In-gram writes when a held-out session is read from its first token (data only, no model; CPU).

scripts/slot_growth.py counts slots in the evaluation windows, which are cut at random places inside sessions.
A served session starts at its first token, and the start of a session contains more new text than its middle,
so this script counts slots in prefixes of whole held-out trajectories instead. A slot is one distinct
(cue length, N-gram) that has been written: an N-gram of the paper's cue lengths that is followed by a token.
The first token of a trajectory is not part of any cue, as in src.schedule (position 0 is BOS there); the count
is checked against src.schedule.build_schedule on a few trajectories.

    .venv/bin/python scripts/slots_from_start.py
Writes results/slots_from_start.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import agentic  # noqa: E402
from src.schedule import build_schedule  # noqa: E402

CUES = (1, 2, 4, 8, 16, 32)
LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
MULT = np.random.default_rng(12345).integers(1, 2**63, size=32, dtype=np.uint64) | np.uint64(1)


def slots(w):
    """Slots written in the context w, per cue length: distinct N-grams over w[1:] that a token follows."""
    x = w[1:].astype(np.uint64)
    out = []
    for N in CUES:
        if len(x) > N:
            h = np.lib.stride_tricks.sliding_window_view(x, N)[: len(x) - N] @ MULT[:N]
            out.append(len(np.unique(h)))
        else:
            out.append(0)
    return np.array(out)


def main():
    t0 = time.time()
    corpus = agentic.load("validation", "gpt2", max_rows=16384)
    lens = np.asarray(corpus.lengths)
    for j in (0, 1, 2):  # same definition as the training schedule
        w = np.asarray(corpus.get(j))[:16384].astype(np.int64)
        ref = int(build_schedule(w[None], orders=CUES, H=1, mode="growing")["n_slots"][0])
        assert slots(w).sum() == ref, (j, slots(w).sum(), ref)
    per_len, counts = [], {}
    for L in LENGTHS:
        idx = np.flatnonzero(lens >= L)
        c = np.array([slots(np.asarray(corpus.get(int(j)))[:L].astype(np.int64)) for j in idx])
        counts[L] = dict(zip(idx.tolist(), c.sum(1).tolist()))
        r = c.sum(1) / L
        per_len.append(dict(context=L, trajectories=int(len(idx)), slots_mean=float(c.sum(1).mean()),
                            per_token=float(r.mean()), per_token_median=float(np.median(r)),
                            per_token_min=float(r.min()), per_token_max=float(r.max()),
                            per_cue_per_token={str(N): float(v) for N, v in zip(CUES, c.mean(0) / L)}))
        print(f"L = {L:6d}: {len(idx):3d} trajectories, slots per token {r.mean():.3f} "
              f"(median {np.median(r):.2f}, {r.min():.2f} to {r.max():.2f}); by cue length "
              + ", ".join(f"{N}: {v:.3f}" for N, v in zip(CUES, c.mean(0) / L)), flush=True)
    marginal = []
    for a, b in zip(LENGTHS[:-1], LENGTHS[1:]):  # the same trajectories at both lengths
        d = [(counts[b][j] - counts[a][j]) / (b - a) for j in counts[b]]
        marginal.append(dict(start=a, end=b, trajectories=len(d), new_slots_per_token=float(np.mean(d))))
        print(f"new slots per token between {a} and {b}: {np.mean(d):.3f} ({len(d)} trajectories)")
    out = dict(cues=list(CUES), held_out_trajectories=int(len(lens)), by_length=per_len, marginal=marginal)
    Path("results").mkdir(exist_ok=True)
    Path("results/slots_from_start.json").write_text(json.dumps(out, indent=1))
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
