"""Phase 29 confirmation analysis, exactly as pre-registered in docs/PREREG_PHASE29.md.

Seed-paired (seeds 20-27) per-model mean NLL on the 32 held-out 16k windows:
  long  = conf29-<set>-s{sd}   (GDN x 8 + locker P = 1, confirmed cue set)
  short = conf6-a0p1-s{sd}     (same, cues 1-5)
  hyb   = conf6-a2p0-s{sd}     (3:1 hybrid, no overlay)
Primary D1 = long - short; secondary D2 = long - hyb; secondary D2c = D2 after mixing each model with the
free n-gram cache (scripts/cache_mix.py, 2-fold cross-fit). One-sided Wilcoxon signed-rank (H1: < 0),
Holm across the three tests. 95% bootstrap CI over seeds (10k resamples).

Downloads any missing per-token .npz artifact first. Usage:
    python scripts/analyze_phase29.py o12481632
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cache_mix import fit  # noqa: E402

SEEDS = range(20, 28)


def nll_file(key: str) -> Path:
    d = Path("runs") / key
    hits = list(d.glob("*.npz"))
    if hits:
        return hits[0]
    import rhombic

    r = json.load(open(d / "result.json"))
    rid = next(iter(r["runs"]))
    dst = d / f"{rid}.npz"
    rhombic.Artifact.get(r["runs"][rid]["artifacts"][f"{rid}.npz"]).download(dst)
    return dst


def scores(key: str, B, Q) -> tuple[float, float]:
    """(mean NLL, mean NLL mixed with the cache, 2-fold cross-fit)."""
    nll = np.load(nll_file(key))["nll"].astype(np.float64)
    ok = ~np.isnan(nll)
    mixed = np.zeros_like(nll)
    halves = [np.arange(16), np.arange(16, 32)]
    for tr, te in [(halves[0], halves[1]), (halves[1], halves[0])]:
        okt = ok[tr]
        lam = fit(np.exp(-nll[tr][okt]), Q[tr][okt], B[tr][okt], slice(None))
        lt = lam[B[te]]
        mixed[te] = -np.log((1 - lt) * np.exp(-nll[te]) + lt * Q[te] + 1e-12)
    return float(nll[ok].mean()), float(mixed[ok].mean())


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


if __name__ == "__main__":
    tag = sys.argv[1]
    z = np.load("runs/tags/cache16k.npz")
    B, Q = z["bucket"], z["q"]
    rows = {sd: {a: scores(k, B, Q) for a, k in (("long", f"conf29-{tag}-s{sd}"), ("short", f"conf6-a0p1-s{sd}"),
                                                  ("hyb", f"conf6-a2p0-s{sd}"))} for sd in SEEDS}
    print("| seed | long | short | hybrid | long+cache | hybrid+cache |")
    print("|---|---|---|---|---|---|")
    for sd, r in rows.items():
        print(f"| {sd} | {r['long'][0]:.4f} | {r['short'][0]:.4f} | {r['hyb'][0]:.4f} | {r['long'][1]:.4f} | "
              f"{r['hyb'][1]:.4f} |")
    tests = {
        "D1 long - short (primary)": np.array([r["long"][0] - r["short"][0] for r in rows.values()]),
        "D2 long - hybrid": np.array([r["long"][0] - r["hyb"][0] for r in rows.values()]),
        "D2c long - hybrid, both + cache": np.array([r["long"][1] - r["hyb"][1] for r in rows.values()]),
    }
    ps = np.array([wilcoxon(d, alternative="less").pvalue for d in tests.values()])
    adj = holm(ps)
    print()
    for (name, d), p, pa in zip(tests.items(), ps, adj):
        lo, hi = boot_ci(d)
        print(f"{name}: mean {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], {int((d < 0).sum())}/8 negative, "
              f"Wilcoxon p = {p:.4f}, Holm p = {pa:.4f}")
    print(f"\nD1 (short, + cache, reported): "
          f"{np.mean([r['long'][1] - r['short'][1] for r in rows.values()]):+.4f}")
