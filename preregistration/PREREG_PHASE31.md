# Pre-registration: Phase 31, wave B (written 2026-11-09 20:10Z, before any wave-B run)

Plan: docs/PLAN_FINAL_SPEND.md (with the approved amendment). Two settings are chosen by rules stated here
from wave-A runs that are still in flight; no wave-B result can influence them.

## Fixed reference arms (existing)
- conf29-o12481632-s{20..27}: locker recipe (GDN x 8, d256, P = 1, cues 1,2,4,8,16,32, lr 2e-3, 10k steps).
- conf30-hyb54-s{20..27}: 3:1 hybrid, 12 x d384, 54.2M, lr 2e-3, 10k steps.

## Wave-B arms (seeds 20-27 unless stated)
| id | campaign | config |
|---|---|---|
| B-long-hyb | b31-longhyb-s{sd} | 30M 3:1 hybrid (as conf6-a2p0), lr 4e-3, **20,000 steps** (cosine over its own length) |
| B-long-loc | b31-longloc-s{sd} | conf29 recipe, lr 2e-3, **20,000 steps** |
| B-size2 | b31-size2-s{sd} | locker at 12 x d384 (as sd15-a0p1 with LOCKER31 cues), lr 2e-3, 10k steps |
| B-static | b31-static-s{sd} | 3:1 hybrid + static hashed table (LOCKER31 cues and size, writes off), 10k steps, lr by rule S |
| B-size3 | b31-size3-{locker,hybrid}-s{20..23} | 16 x d512, 20k steps (328M tokens), lr per arm by rule T |

**Rule S.** lr for B-static = the lower mean post-eval NLL over seeds 0-1 of a31-static-lr2e3 vs
a31-static-lr4e3 (tie: 2e-3).
**Rule T.** lr per third-size arm = the lower post-eval NLL at seed 0 of a31-size3-<arm>-lr1e3 vs -lr2e3
(tie: 1e-3). Four seeds per arm (20-23); if the measured cost would exceed the $50 cap, three (20-22).

## Tests (one Holm family across all of them; one-sided Wilcoxon signed-rank, H1: locker lower)
| test | difference (seed-paired) | n |
|---|---|---|
| T1 | B-long-loc - B-long-hyb (gap at 20k steps, both arms) | 8 |
| T2 | conf29 - B-long-hyb (locker at 10k vs hybrid at 20k: equal wall-clock for step-time ratio <= 2) | 8 |
| T3 | conf29 - B-static (parameter-matched Qwen-style design) | 8 |
| T4 | B-size2 - conf30 (second size) | 8 |
| T5 | B-size3 locker - hybrid | 4 (or 3) |
With n = 4 the smallest one-sided Wilcoxon p is 0.0625, so T5 cannot pass alpha = 0.05 by construction;
it is reported as an effect with its CI and sign count, not as a test, and is excluded from the Holm family
(family T1-T4).

Also reported, not tested: each comparison after mixing both models with the free n-gram cache
(scripts/cache_mix.py procedure), medians beside means, the gap by training step for T1 (2k-step evals),
and B-long-loc - conf29 (what 10k more steps buy locker). 95% bootstrap CIs over seeds (10k resamples).
Script: scripts/analyze_phase31.py, committed before results. No seed is dropped; failed runs are relaunched
with the same seed. Deviations go to DEVIATIONS.md.
