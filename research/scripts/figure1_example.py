"""Counts behind the example of Figure 1b (data only, no model; CPU).

The figure shows the cue "tmp", "_" (GPT-2 tokens) followed by "folder" in a held-out SWE-rebench trajectory
(index 186 of the held-out split; the text is `self.tmp_folder1`, `self.tmp_folder2`). This script counts what
follows that cue (a) in the first 16,384 tokens of that trajectory, (b) in the first 16,384 tokens of a fixed
sample of 2,000 training trajectories, and (c) in the other held-out trajectories.

    .venv/bin/python scripts/figure1_example.py
Writes results/figure1_example.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import agentic  # noqa: E402
from src.tagger import token_strings  # noqa: E402

CUE, TRAJ, L, N_TRAIN = ("tmp", "_"), 186, 16384, 2000


def follow(corpus, idx, a, b, ts):
    tot, sessions = Counter(), 0
    for j in idx:
        x = np.asarray(corpus.get(int(j)))[:L].astype(np.int64)
        m = np.flatnonzero((x[:-2] == a) & (x[1:-1] == b))
        if len(m):
            sessions += 1
            tot.update(ts[int(v)] for v in x[m + 2])
    return tot, sessions


def main():
    ts = token_strings("gpt2")
    a, b = ts.index(CUE[0]), ts.index(CUE[1])
    held = agentic.load("validation", "gpt2", max_rows=16384)
    train = agentic.load("train", "gpt2", max_rows=16384)
    own, _ = follow(held, [TRAJ], a, b, ts)
    w = np.asarray(held.get(TRAJ))[:L].astype(np.int64)
    first = int(np.flatnonzero((w[:-2] == a) & (w[1:-1] == b))[0]) + 2
    pick = np.random.default_rng(0).choice(len(train.lengths), size=N_TRAIN, replace=False)
    tr, tr_s = follow(train, pick, a, b, ts)
    oth, oth_s = follow(held, [j for j in range(len(held.lengths)) if j != TRAJ], a, b, ts)
    out = dict(cue=list(CUE), trajectory=TRAJ, window=L,
               text_at_first_occurrence="".join(ts[int(x)] for x in w[first - 8:first + 6]),
               in_trajectory=dict(uses=sum(own.values()), continuations=dict(own)),
               training_sample=dict(trajectories=N_TRAIN, with_cue=tr_s, uses=sum(tr.values()),
                                    continuations=dict(tr.most_common(10)), folder=tr["folder"]),
               other_held_out=dict(trajectories=len(held.lengths) - 1, with_cue=oth_s, uses=sum(oth.values()),
                                   continuations=dict(oth.most_common(10))))
    Path("results").mkdir(exist_ok=True)
    Path("results/figure1_example.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
