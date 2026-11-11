"""Statistics and tables (plan.md §8).

python -m src.analyze runs/<campaign>[,runs/<campaign2>...] [--metric acc] [--focus ovl,locker_gdn]

Reads every <run_id>.jsonl in the given folders and groups runs by arm label: the run_id with its
"-s<seed>" suffix removed. Then it prints:
- per-group final metric: mean, bootstrap 95% CI, per-seed values;
- escape-step summary (first eval >= 0.8; censored at the last step): median and fraction escaped;
- one-sided Mann-Whitney U tests (focus > other), Holm-corrected within each focus arm,
  with effect size = difference in means + bootstrap 95% CI.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu


def load_runs(folders: list[str]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for folder in folders:
        for f in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
            name = os.path.basename(f)[:-6]
            if name.startswith("smoke-"):
                continue
            lines = [json.loads(l) for l in open(f) if l.strip()]
            final = next((r for r in lines if r.get("final")), None)
            evals = [r for r in lines if "acc" in r and "step" in r]
            if not final or not evals:
                continue
            m = re.match(r"(.+)-s(\d+)$", name)
            label, seed = (m.group(1), int(m.group(2))) if m else (name, 0)
            groups[label].append(dict(seed=seed, final=final, evals=evals, header=lines[0]))
    return groups


def bootstrap_ci(x, fn=np.mean, n=10_000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = [fn(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def diff_ci(a, b, n=10_000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def escape_summary(runs, thresh=0.8):
    steps, escaped = [], []
    for r in runs:
        e = next((ev["step"] for ev in r["evals"] if ev["acc"] >= thresh), None)
        escaped.append(e is not None)
        steps.append(e if e is not None else r["evals"][-1]["step"])  # censored at budget
    s = np.array(steps)
    esc = np.array(escaped)
    med = float(np.median(s[esc])) if esc.any() else float("nan")
    return dict(frac_escaped=float(esc.mean()), median_escape=med, censored_at=int(s.max()))


def report(groups, metric="acc", focus=("ovl", "locker_gdn")):
    vals = {g: [r["final"].get(metric) for r in rs if r["final"].get(metric) is not None] for g, rs in groups.items()}
    print(f"\n## Final {metric} per arm (mean, bootstrap 95% CI, seeds)")
    print("| arm | n | mean | 95% CI | per-seed | escaped | median escape |")
    print("|---|---|---|---|---|---|---|")
    for g in sorted(vals):
        v = vals[g]
        if not v:
            continue
        lo, hi = bootstrap_ci(v)
        es = escape_summary(groups[g])
        print(f"| {g} | {len(v)} | {np.mean(v):.3f} | [{lo:.3f}, {hi:.3f}] | "
              f"{' '.join(f'{x:.2f}' for x in v)} | {es['frac_escaped']:.0%} | {es['median_escape']} |")
    for f in focus:
        fk = [g for g in vals if g == f or g.startswith(f + "-")]
        for fg in fk:
            others = {g: v for g, v in vals.items() if g != fg and v and not g.startswith(fg)}
            if not others:
                continue
            p = {g: mannwhitneyu(vals[fg], v, alternative="greater").pvalue for g, v in others.items()}
            ph = holm(p)
            print(f"\n## {fg} > other arms (one-sided Mann-Whitney U, Holm-corrected)")
            print("| vs | diff in means | 95% CI | p | p (Holm) |")
            print("|---|---|---|---|---|")
            for g in sorted(others, key=lambda k: ph[k]):
                lo, hi = diff_ci(vals[fg], others[g])
                print(f"| {g} | {np.mean(vals[fg]) - np.mean(others[g]):+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                      f"{p[g]:.2g} | {ph[g]:.2g} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", help="comma-separated run folders")
    ap.add_argument("--metric", default="acc")
    ap.add_argument("--focus", default="ovl,locker_gdn")
    a = ap.parse_args()
    report(load_runs(a.folders.split(",")), a.metric, tuple(a.focus.split(",")))


if __name__ == "__main__":
    main()
