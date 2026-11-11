"""Tag the dose grid's eval windows (same construction as train.py: eval_seed 987654, batch 1, 32 x 16k
windows from the held-out repos of the 16k-trajectory pool) and report each tag's share of tokens.
Saves the tags to runs/tags/<name>.npz for per-tag losses later (scripts/tag_losses.py)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tagger import TAGS, tag_window, token_strings  # noqa: E402
from src.tasks import make_task  # noqa: E402
from src.train import with_lmax  # noqa: E402


def eval_windows(task, n, batch, seed=987654):
    _, fn = make_task({**with_lmax(task), "split": "validation"})
    rng = np.random.default_rng(seed)
    out = []
    for i in range(0, n, batch):
        b = fn(min(batch, n - i), rng)
        out.extend(list(b["targets"]))
    return out


if __name__ == "__main__":
    L, n, batch, name = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    task = dict(name="agentic", L_max=L, tokenizer="gpt2", max_rows=16384)
    strs = token_strings("gpt2")
    wins = eval_windows(task, n, batch)
    t0 = time.time()
    tags = {k: [] for k in TAGS}
    for w in wins:
        w = np.where(w < 0, 0, w)  # position 0 is IGNORE (unscored)
        tg = tag_window(w, strs)
        for k in TAGS:
            v = tg[k].copy()
            v[0] = False
            tags[k].append(v)
    print(f"{len(wins)} windows x {L} tagged in {time.time() - t0:.0f} s")
    total = sum(len(w) - 1 for w in wins)
    for k in TAGS:
        print(f"  {k:13s} {sum(int(v.sum()) for v in tags[k]) / total:7.2%}")
    out = Path("runs/tags"); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", **{k: np.stack(v) for k, v in tags.items()})
    print("saved", out / f"{name}.npz")
