# Pre-registration: Phase 32, long training (written 2026-11-10, before any run; authorized by the user, ~$42)

## Question
Framing: locker helps a hybrid use fewer attention layers at equal quality. Does that still hold when the
models are trained well past compute-optimal? At 10k steps (sd14/sd29, seeds 0-1): 3:1 hybrid 0.722,
7:1 hybrid 0.995, 7:1 + locker 0.607.

## Arms (80,000 steps x 16k tokens = 1.31B tokens, about one pass over the 16,384-trajectory pool; cosine
schedule over the full length; seeds 20-27; everything else as the sd14/sd29 grid cells)
| id | campaign | config | lr |
|---|---|---|---|
| H3 | p32-a2p0-s{sd} | 3:1 hybrid (attention at blocks 4 and 8), no locker | 4e-3 |
| L7 | p32-a1p1-s{sd} | 7:1 hybrid (attention at block 4) + locker after block 1, cues 1,2,4,8,16,32 | 2e-3 |
| H7 | p32-a1p0-s{sd} | 7:1 hybrid, no locker | 4e-3 |
LRs are the grid's (tuned at 10k steps) and are not re-tuned for 80k; stated as a limitation.

## Tests (seed-paired, n = 8, one-sided Wilcoxon signed-rank, H1: L7 lower; Holm across the two)
- **Primary T1:** L7 - H3 (7:1 with locker vs the standard 3:1 without).
- **T2:** L7 - H7 (locker is what makes 7:1 work).
Also reported: both with the free n-gram cache (scripts/cache_mix.py procedure), medians, per-type gaps
(scripts/gap_by_type.py definitions), and 95% bootstrap CIs over seeds (10k resamples).

## How each primary outcome is read (fixed now)
- 95% CI of T1 below zero: 7:1 + locker beats 3:1 at 1.3B tokens; locker saves an attention layer.
- CI includes zero: parity at 1.3B tokens; locker still lets a hybrid drop to 7:1 at equal quality.
- CI above zero: 3:1 beats 7:1 + locker at 1.3B tokens; reported as such, and the paper's claim is limited
  to shorter training.

Script: scripts/analyze_phase32.py, committed before results. No seed is dropped; failed runs are
relaunched with the same seed. Deviations go to DEVIATIONS.md.
