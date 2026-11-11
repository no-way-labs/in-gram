"""Phase 31 (wave B) analysis exactly as pre-registered in docs/PREREG_PHASE31.md.
T1-T4: one-sided Wilcoxon (H1: locker lower), one Holm family; T5 (n = 4) reported as effect + CI + signs.
Also reported: the same differences with both models mixed with the free n-gram cache, medians, the T1 gap by
training step, and longloc - conf29.   python scripts/analyze_phase31.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_phase29 import boot_ci, holm, scores  # noqa: E402

S8 = range(20, 28)


def curve(key):
    out = {}
    for line in open(glob.glob(f"runs/{key}/*.jsonl")[0]):
        d = json.loads(line)
        if "step" in d and d.get("loss") is not None:
            out[d["step"]] = d["loss"]
    return out


def report(name, d, test=True):
    lo, hi = boot_ci(d)
    p = wilcoxon(d, alternative="less").pvalue if test else None
    return dict(name=name, mean=d.mean(), median=float(np.median(d)), lo=lo, hi=hi, neg=int((d < 0).sum()),
                n=len(d), p=p)


if __name__ == "__main__":
    z = np.load("runs/tags/cache16k.npz")
    B, Q = z["bucket"], z["q"]
    sc = {}

    def s(key):
        if key not in sc:
            sc[key] = scores(key, B, Q)
        return sc[key]

    pairs = {
        "T1 longloc - longhyb (20k vs 20k)": [(f"b31-longloc-s{i}", f"b31-longhyb-s{i}") for i in S8],
        "T2 conf29 (10k) - longhyb (20k)": [(f"conf29-o12481632-s{i}", f"b31-longhyb-s{i}") for i in S8],
        "T3 conf29 - static-table hybrid": [(f"conf29-o12481632-s{i}", f"b31-static-s{i}") for i in S8],
        "T4 size2 locker - conf30 hybrid": [(f"b31-size2-s{i}", f"conf30-hyb54-s{i}") for i in S8],
    }
    t5 = [(f"b31-size3-locker-s{i}", f"b31-size3-hybrid-s{i}") for i in range(20, 24)
          if Path(f"runs/b31-size3-locker-s{i}/result.json").exists() and Path(f"runs/b31-size3-hybrid-s{i}/result.json").exists()]
    rows, rows_c = [], []
    for name, pr in pairs.items():
        rows.append(report(name, np.array([s(a)[0] - s(b)[0] for a, b in pr])))
        rows_c.append(report(name + " (+cache)", np.array([s(a)[1] - s(b)[1] for a, b in pr]), test=False))
    adj = holm(np.array([r["p"] for r in rows]))
    print("| test | mean [95% CI] | median | negative | Wilcoxon p | Holm p |\n|---|---|---|---|---|---|")
    for r, pa in zip(rows, adj):
        print(f"| {r['name']} | {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['median']:+.4f} | "
              f"{r['neg']}/{r['n']} | {r['p']:.4f} | {pa:.4f} |")
    if t5:
        d = np.array([s(a)[0] - s(b)[0] for a, b in t5])
        r = report("T5 size3 locker - hybrid (reported, not tested)", d, test=False)
        print(f"| {r['name']} | {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['median']:+.4f} | "
              f"{r['neg']}/{r['n']} | - | - |")
        dc = np.array([s(a)[1] - s(b)[1] for a, b in t5])
        print(f"T5 with cache: {dc.mean():+.4f}")
    print("\nWith the free n-gram cache on both sides (reported):")
    for r in rows_c:
        print(f"- {r['name']}: {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}], median {r['median']:+.4f}, {r['neg']}/{r['n']}")
    print("\nT1 gap by training step (mean / median over seeds):")
    gaps = {}
    for i in S8:
        a, b = curve(f"b31-longloc-s{i}"), curve(f"b31-longhyb-s{i}")
        for st in sorted(set(a) & set(b)):
            gaps.setdefault(st, []).append(a[st] - b[st])
    for st, v in sorted(gaps.items()):
        print(f"  {st}: {np.mean(v):+.4f} / {np.median(v):+.4f} (n = {len(v)})")
    d = np.array([s(f"b31-longloc-s{i}")[0] - s(f"conf29-o12481632-s{i}")[0] for i in S8])
    print(f"\nlongloc (20k) - conf29 (10k): {d.mean():+.4f} [{boot_ci(d)[0]:+.4f}, {boot_ci(d)[1]:+.4f}]")
