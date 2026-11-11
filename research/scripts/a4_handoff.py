"""PREREG_PHASE27 A4: induction share of far attention (far_ind / far, averaged over heads and layers)
in the A = 2 and A = 4 cells, P = 0 vs 1, 2, 4, per seed. Also reports the far-attention mass itself.
Reads runs/sd14-a{A}p{P}-s{0,1}/*.post_eval.json.  python scripts/a4_handoff.py
"""
import glob
import json

import os

import numpy as np

GRID = os.environ.get("GRID", "sd14")  # overlay cells (P >= 1) from this grid; P = 0 always sd14

print("| A | P | far_ind / far (s0, s1) | mean | far mass (mean) |")
print("|---|---|---|---|---|")
for A in (2, 4):
    for P in (0, 1, 2, 4):
        shares, mass = [], []
        for f in sorted(glob.glob(f"runs/{GRID if P else 'sd14'}-a{A}p{P}-s[01]/*.post_eval.json")):
            m = next(iter(json.load(open(f))["models"].values()))["attn"]
            far = np.concatenate([np.asarray(v["far"]) for v in m.values()])
            ind = np.concatenate([np.asarray(v["far_ind"]) for v in m.values()])
            shares.append(float(np.mean(ind / far)))
            mass.append(float(far.mean()))
        print(f"| {A} | {P} | " + ", ".join(f"{s:.3f}" for s in shares)
              + f" | {np.mean(shares):.3f} | {np.mean(mass):.3f} |")
