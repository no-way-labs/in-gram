"""Pre-registered phase-27 analysis (docs/PREREG_PHASE27.md): Part A1/A2 (dose grid iso-loss),
Part B (conf6: A0P1 vs A2P0), Part C (conf7: ctxkey K=4 vs additive; K=8 secondary).
Metric: mean held-out NLL at step 10,000 (post_eval mean_nll when present, else the step-10000 eval
loss in the run log; the two agree). Missing runs are reported as missing, never imputed.
Usage: python scripts/analyze_phase27.py > results/phase27.md"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
AS, PS = (0, 1, 2, 4), (0, 1, 2, 4)
GRID = __import__("os").environ.get("GRID", "sd14")  # overlay cells (P >= 1) from this grid; P = 0 cells always sd14


def gkey(A, P, s):
    return f"{GRID if P else 'sd14'}-a{A}p{P}-s{s}"


def loss(key):
    d = RUNS / key
    for pe in d.glob("*.post_eval.json"):
        m = json.loads(pe.read_text())["models"]
        return float(next(iter(m.values()))["mean_nll"])
    for lg in d.glob("*.jsonl"):
        last = None
        for line in lg.read_text().splitlines():
            r = json.loads(line)
            if r.get("step") == 10000 and "loss" in r:
                last = r["loss"]
        if last is not None:
            return float(last)
    return None


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


def paired(name, a_keys, b_keys, label):
    a = [loss(k) for k in a_keys]
    b = [loss(k) for k in b_keys]
    ok = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    print(f"\n## {name}\n")
    print(f"| seed pair | {label[0]} | {label[1]} | diff |\n|---|---|---|---|")
    for ka, x, y in zip(a_keys, a, b):
        s = ka.rsplit("-s", 1)[1]
        print(f"| s{s} | {x if x is None else f'{x:.4f}'} | {y if y is None else f'{y:.4f}'} | "
              f"{'' if x is None or y is None else f'{x - y:+.4f}'} |")
    if len(ok) < len(a_keys):
        print(f"\nMissing pairs: {len(a_keys) - len(ok)} of {len(a_keys)} (test not run until complete).")
        if len(ok) < 2:
            return
    d = np.array([x - y for x, y in ok])
    w = stats.wilcoxon(d, alternative="less")
    t = stats.ttest_rel([x for x, _ in ok], [y for _, y in ok], alternative="less")
    lo, hi = boot_ci(d)
    print(f"\nn = {len(d)} pairs. Mean diff {d.mean():+.4f} nats (95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]); "
          f"{int((d < 0).sum())} of {len(d)} negative. One-sided Wilcoxon signed-rank p = {w.pvalue:.4g}; "
          f"paired t p = {t.pvalue:.4g}. Means: {label[0]} {np.mean([x for x, _ in ok]):.4f}, "
          f"{label[1]} {np.mean([y for _, y in ok]):.4f}.")


def a_star(curve, ref):
    """Smallest A (linear interpolation over AS) with loss <= ref; curve = losses at AS."""
    if curve[0] <= ref:
        return 0.0, True
    for (a0, l0), (a1, l1) in zip(zip(AS, curve), zip(AS[1:], curve[1:])):
        if l1 <= ref:
            return a0 + (l0 - ref) / (l0 - l1) * (a1 - a0), False
    return None, False


def part_a():
    print("\n## Part A2: loss surface (mean held-out NLL, nats/token; seed 0 / seed 1 / mean)\n")
    print("| A \\ P | " + " | ".join(f"P={p}" for p in PS) + " |\n" + "|---" * (len(PS) + 1) + "|")
    L = {}
    for A in AS:
        cells = []
        for P in PS:
            v = [loss(gkey(A, P, s)) for s in (0, 1)]
            L[A, P] = v
            have = [x for x in v if x is not None]
            m = f"{np.mean(have):.4f}" if have else "-"
            cells.append(" / ".join("-" if x is None else f"{x:.4f}" for x in v) + f" / **{m}**")
        print(f"| A={A} | " + " | ".join(cells) + " |")
    g8 = [loss(f"sd14-a0p0-lr8e3-s{s}") for s in (0, 1)]
    print(f"\nPure GDN at lr 8e-3 (no overlay): {g8}")
    x8 = [loss(f"sd14x-a8p0-s{s}") for s in (0, 1)]
    print(f"All-attention (A = 8, sd14x): {x8}")

    ref_v = L[2, 0]
    if None in ref_v:
        print("\nA1 not computed: reference cell A2P0 incomplete.")
        return
    ref = float(np.mean(ref_v))
    print(f"\n## Part A1: iso-loss attention A*(P) vs L_ref = A2P0 seed mean = {ref:.4f}\n")
    print("| P | A*(P) seed mean | A*(P) s0 | A*(P) s1 | layers saved (2 - A*) | margin L_ref - L(A=0,P) |")
    print("|---|---|---|---|---|---|")
    per_seed = {0: [], 1: []}
    for P in PS[1:]:
        if any(None in L[A, P] for A in AS):
            miss = [A for A in AS if None in L[A, P]]
            print(f"| {P} | incomplete (A = {miss} missing) | | | | |")
            continue
        mean_curve = [float(np.mean(L[A, P])) for A in AS]
        am, le = a_star(mean_curve, ref)
        row = []
        for s in (0, 1):
            a, le_s = a_star([L[A, P][s] for A in AS], ref)
            per_seed[s].append(a)
            row.append(("<= 0" if le_s else f"{a:.2f}") if a is not None else "> 4")
        mstr = ("<= 0" if le else f"{am:.2f}") if am is not None else "> 4"
        saved = f">= 2 (A* <= 0)" if le else (f"{2 - am:.2f}" if am is not None else "-")
        print(f"| {P} | {mstr} | {row[0]} | {row[1]} | {saved} | {ref - mean_curve[0]:+.4f} |")
    for s in (0, 1):
        v = per_seed[s]
        if len(v) == 3 and None not in v:
            mono = all(b <= a + 1e-9 for a, b in zip(v, v[1:]))
            print(f"\nSeed {s}: A*(P) for P = 1, 2, 4 = {[round(x, 3) for x in v]}; non-increasing: {mono}")


def main():
    print("# Phase 27 results (pre-registered analysis, docs/PREREG_PHASE27.md)")
    part_a()
    paired("Part B: attention-free + overlay (A0P1) vs 3:1 hybrid (A2P0), seeds 20-27",
           [f"conf6-a0p1-s{s}" for s in range(20, 28)], [f"conf6-a2p0-s{s}" for s in range(20, 28)],
           ("A0P1", "A2P0"))
    paired("Part C: locker_a (ctxkey, K = 4) vs additive, seeds 13-20",
           [f"conf7-ctxkey-k4-s{s}" for s in range(13, 21)], [f"conf7-additive-s{s}" for s in range(13, 21)],
           ("ctxkey K=4", "additive"))
    paired("Part C secondary (not tested by pre-registration; reported): ctxkey K = 8 vs additive",
           [f"conf7-ctxkey-k8-s{s}" for s in range(13, 21)], [f"conf7-additive-s{s}" for s in range(13, 21)],
           ("ctxkey K=8", "additive"))


if __name__ == "__main__":
    main()
