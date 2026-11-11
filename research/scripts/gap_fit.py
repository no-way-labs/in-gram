"""Exploratory, Discussion-only: how the locker-vs-hybrid gap moves with training length. Two views.
(1) Endpoints: final gaps of the 10k runs (conf29 vs conf6-a2p0) and 20k runs (b31-longloc vs -longhyb),
    seeds 20-27 -> per-doubling retention r = gap20k / gap10k, and a naive power-law projection.
(2) Within the 20k runs: gap(t) = c + a * t^-b fitted to the 2k-step evals (cosine schedule: intermediate points
    are not finished runs, so this describes the curve and is not a forecast). Bootstrap over seeds.
    python scripts/gap_fit.py
"""
import glob
import json

import numpy as np
from scipy.optimize import curve_fit


def final(k):
    r = json.load(open(f"runs/{k}/result.json")); rid = next(iter(r["runs"]))
    return r["runs"][rid]["post_eval"]["models"][rid]["mean_nll"]


def curve(k):
    out = {}
    for line in open(glob.glob(f"runs/{k}/*.jsonl")[0]):
        d = json.loads(line)
        if "step" in d and d.get("loss") is not None:
            out[d["step"]] = d["loss"]
    return out


S = range(20, 28)
g10 = np.array([final(f"conf29-o12481632-s{s}") - final(f"conf6-a2p0-s{s}") for s in S])
g20 = np.array([final(f"b31-longloc-s{s}") - final(f"b31-longhyb-s{s}") for s in S])
r = g20.mean() / g10.mean()
rng = np.random.default_rng(0)
rb = [g20[i].mean() / g10[i].mean() for i in rng.integers(0, 8, (10000, 8))]
print(f"(1) final gap 10k {g10.mean():+.4f}, 20k {g20.mean():+.4f}; retained per doubling r = {r:.2f} "
      f"[{np.percentile(rb, 2.5):.2f}, {np.percentile(rb, 97.5):.2f}]")
for mult in (2, 8, 32, 128):  # 20k x mult steps
    print(f"    naive projection at {20 * mult}k steps ({0.328 * mult:.1f}B tokens): {g20.mean() * r ** np.log2(mult):+.4f}")

C = {s: (curve(f"b31-longloc-s{s}"), curve(f"b31-longhyb-s{s}")) for s in S}
steps = np.array(sorted(set.intersection(*[set(a) & set(b) for a, b in C.values()])), float)
G = np.array([[C[s][0][t] - C[s][1][t] for t in steps] for s in S])
f = lambda t, c, a, b: c + a * (t / 1000.0) ** (-b)  # noqa: E731


def fit(g):
    p, _ = curve_fit(f, steps, g, p0=(-0.02, -2.0, 1.0), maxfev=20000)
    return p


p = fit(G.mean(0))
cs = []
for i in rng.integers(0, 8, (2000, 8)):
    try:
        cs.append(fit(G[i].mean(0))[0])
    except RuntimeError:
        pass
print(f"(2) within-run fit gap = c + a (t/1k)^-b: c = {p[0]:+.4f} [{np.percentile(cs, 2.5):+.4f}, "
      f"{np.percentile(cs, 97.5):+.4f}], a = {p[1]:+.3f}, b = {p[2]:.2f} ({len(cs)} bootstrap fits)")
