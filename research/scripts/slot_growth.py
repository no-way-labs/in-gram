"""How In-gram's per-session slot count grows with context (data only, no model; CPU).

A slot is one distinct (cue length, hashed N-gram) written in a session, exactly as the training
schedule counts it (src.schedule.build_schedule, growing mode, n_slots). For the paper's cue lengths
{1, 2, 4, 8, 16, 32} we count slots in prefixes of held-out agentic windows:
  * the 32 evaluation windows of 16,384 tokens (the paper's eval set), at 1k .. 16k;
  * longer held-out windows (64k, 256k) drawn the same way, with the share that had to join several
    trajectories because the sampled one was shorter than the window.
Prints slots per token and the marginal rate (new slots per new token) between successive lengths.

    .venv/bin/python scripts/slot_growth.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tag_eval_set import eval_windows  # noqa: E402
from src.schedule import build_schedule  # noqa: E402

ORDERS = (1, 2, 4, 8, 16, 32)


def slots(tok, orders=ORDERS):
    return int(build_schedule(np.asarray(tok, dtype=np.int64)[None], orders=orders, H=1, mode="growing")["n_slots"][0])


def growth(wins, lengths):
    out = {}
    for L in lengths:
        out[L] = np.array([slots(w[:L]) for w in wins])
    return out


def report(name, wins, lengths):
    g = growth(wins, lengths)
    print(f"\n## {name} ({len(wins)} windows)")
    print("| context | slots (mean) | slots / token | min - max over windows | new slots per new token since previous row |")
    print("|---|---|---|---|---|")
    prev = None
    rows = []
    for L in lengths:
        m = g[L].mean()
        marg = "" if prev is None else f"{(m - prev[1]) / (L - prev[0]):.2f}"
        print(f"| {L} | {m:.0f} | {m / L:.2f} | {g[L].min() / L:.2f} - {g[L].max() / L:.2f} | {marg} |")
        rows.append(dict(context=L, slots_mean=float(m), per_token=float(m / L), per_token_min=float(g[L].min() / L),
                         per_token_max=float(g[L].max() / L)))
        prev = (L, m)
    return rows


if __name__ == "__main__":
    out = {}
    t0 = time.time()
    task = dict(name="agentic", L_max=16384, tokenizer="gpt2", max_rows=16384)
    wins = [np.where(w < 0, 0, w) for w in eval_windows(task, 32, 1)]
    out["eval16k"] = report("Evaluation windows, 16,384 tokens", wins, [1024, 2048, 4096, 8192, 16384])
    per_order = {N: np.mean([slots(w, (N,)) for w in wins]) / 16384 for N in ORDERS}
    print("slots per token by cue length at 16k: " + ", ".join(f"N={N}: {v:.2f}" for N, v in per_order.items()))
    out["per_order_16k"] = {str(k): float(v) for k, v in per_order.items()}
    for L, n in ((65536, 16), (262144, 8)):
        if "--long" not in sys.argv:
            break
        task = dict(name="agentic", L_max=L, tokenizer="gpt2", max_rows=16384)
        wl = [np.where(w < 0, 0, w) for w in eval_windows(task, n, 1)]
        out[f"long{L}"] = report(f"Held-out windows, {L} tokens", wl, [16384, 65536] + ([262144] if L > 65536 else []))
    Path("results").mkdir(exist_ok=True)
    if "--long" in sys.argv:
        Path("results/slot_growth.json").write_text(json.dumps(out, indent=1))
    print(f"\n{time.time() - t0:.0f} s")
