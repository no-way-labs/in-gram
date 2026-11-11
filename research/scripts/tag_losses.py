"""Per-tag mean NLL for each model: join runs/tags/<tags>.npz (scripts/tag_eval_set.py) with per-token
NLL files (src/nll_dump.py output, downloaded into a directory). Same windows, same order.

    python scripts/tag_losses.py runs/tags/eval16k.npz runs/nll/sd14   # -> markdown table
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tagger import TAGS  # noqa: E402


def table(tag_path, nll_dir):
    tags = np.load(tag_path)
    rows = []
    for f in sorted(Path(nll_dir).glob("*.npz")):
        nll = np.load(f)["nll"].astype(np.float32)
        assert nll.shape == tags[TAGS[0]].shape, (f.name, nll.shape, tags[TAGS[0]].shape)
        ok = ~np.isnan(nll)
        row = {"model": f.stem, "all": float(nll[ok].mean())}
        for k in TAGS:
            m = tags[k] & ok
            row[k] = float(nll[m].mean()) if m.any() else float("nan")
        rows.append(row)
    share = {k: float(tags[k].mean()) for k in TAGS}
    return rows, share


if __name__ == "__main__":
    rows, share = table(sys.argv[1], sys.argv[2])
    cols = ["all", *TAGS]
    print("| model | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    print("| share of tokens | 100% | " + " | ".join(f"{share[k]:.1%}" for k in TAGS) + " |")
    for r in rows:
        print(f"| {r['model']} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
