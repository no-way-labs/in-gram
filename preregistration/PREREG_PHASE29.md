# Pre-registration: Phase 29, long cues (written 2026-11-09, before any seed 20-27 long-cue run)

## Candidate selection (dev seeds 0-1 only)
Cue set for confirmation = the lower mean post-eval NLL (seeds 0-1, GDN x 8 + locker P = 1) of
{1,2,3,5,8,16} (Phase 28c: 0.6111 / 0.6112) and {1,2,4,8,16,32} (Phase 29 pilot, running). The P = 2
pilot arm informs future work only; confirmation stays at P = 1 so it pairs with conf6.

## Confirmation runs
`conf29-<set>-s{20..27}`: GDN x 8 + additive locker, P = 1, the chosen cue set; everything else as
conf6-a0p1 (lr 2e-3, 10k steps x 16k tokens, same post-eval on the 32 held-out 16k windows).
Controls are the existing conf6 runs on the same seeds: conf6-a0p1-s20..27 (cues 1-5) and
conf6-a2p0-s20..27 (3:1 hybrid, no overlay). Configurations verified identical except seed.

## Analyses (seed-paired, n = 8)
- **Primary:** D1 = NLL(long cues) - NLL(cues 1-5). One-sided Wilcoxon signed-rank, H1: D1 < 0, alpha 0.05.
  Report the mean, the 8 differences and a 95% bootstrap CI over seeds.
- **Secondary 1:** D2 = NLL(long cues) - NLL(3:1 hybrid), same test.
- **Secondary 2 (cache baseline):** the same D1 and D2 after mixing each model with the free n-gram
  cache (`scripts/cache_mix.py`, 2-fold cross-fit, per model). Report; test D2-with-cache one-sided.
- Holm correction across the three tests (primary, secondary 1, secondary 2 with cache).
Expectations: D1 about -0.02, D2 about -0.11, D2 with cache about -0.035 (dev seeds).
No result changes this file; deviations go to DEVIATIONS.md.

## Addendum (2026-11-09, before any confirmation run): grid rerun, authorized by the user
If the primary test passes (Holm-adjusted), rerun the 24 locker cells of the sd14 dose grid with the
confirmed cue set (`sd29-a{0,1,2,4}p{1,2,4}-s{0,1}`, ~$10). P = 0 cells are reused. The paper then uses one
recipe throughout: the dose grid and analyses A1, A3, A4 (docs/PREREG_PHASE27.md definitions, unchanged)
on sd29 + the sd14 P = 0 cells; the removing-attention comparison uses conf29 vs conf6-a2p0 (seeds
20-27); the original cues-1-5 confirmation (conf6) moves to the appendix. If the primary test fails,
nothing is rerun and the paper stays on cues 1-5. Either way the recipe is frozen after this.
