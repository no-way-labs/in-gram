# Pre-registration: Phase 30, parameter-matched hybrid (written 2026-11-09, before any run)

## Question
Does the confirmed locker model (conf29: GDN x 8, d256, locker P = 1, cues 1,2,4,8,16,32; 56.0M
parameters, 25.2M of them in the hashed table) still beat a 3:1 hybrid of the same total size?

## Arms
- `conf30-hyb54-s{20..27}`: 3:1 hybrid at the second size (12 blocks x d384, mixer [gdn, gdn, gdn, attn] x 3,
  RoPE attention + conv as in sd15-a3p0), 54.2M parameters (15.6M outside embedding and head, against
  5.1M for the locker model). lr 2e-3, the better of {2e-3, 4e-3} on dev seeds 0-1 (sd15: 0.713 vs 0.745).
  Everything else as conf29 (10k steps x 16k tokens, same data, same post-eval on the 32 held-out windows).
- Comparison arm: the existing conf29-o12481632-s{20..27}. Seeds 20-27 were never used to choose anything
  in either arm.

## Analyses (seed-paired, n = 8, one-sided Wilcoxon signed-rank, H1: locker lower)
- **Primary:** D3 = NLL(conf29) - NLL(conf30-hyb54).
- **Secondary:** D3c = the same after mixing each model with the free n-gram cache
  (scripts/cache_mix.py procedure, 2-fold cross-fit).
- Holm across the two. 95% bootstrap CI over seeds (10k resamples). Script: scripts/analyze_phase30.py,
  committed before results.
Expectation from dev seeds: D3 about -0.10; the hybrid's seed spread at this size is large (0.677 / 0.748).
Any failed run is relaunched with the same seed; no seed is dropped. Deviations go to DEVIATIONS.md.
