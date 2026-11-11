# Pre-registration: dose grid analysis and two confirmatory tests (written 2026-11-08, before any grid result)

Nothing in this file may change after the dose grid's first result is read. Deviations go in
`DEVIATIONS.md` with the reason and the date.

## Common setup (all three parts)
- Data: SWE-rebench trajectories, 16k-trajectory pool (`SD_TASK_BIG`), repo-disjoint held-out split.
- Model: 8 blocks, d_model 256, GPT-2 tokens; GDN mixers except the attention blocks named below.
- Attention blocks: full causal softmax attention with RoPE and a size-4 short conv (`ATT["attn"]`),
  at block 4 (one layer) or blocks 4 and 8 (two layers) or blocks 2, 4, 6, 8 (four layers).
- Overlay ("write path"): additive rule (`ADD`), shared table, placements after blocks [1] / [1, 5] /
  [1, 3, 5, 7] for P = 1 / 2 / 4.
- Training: 10,000 steps x 1 window x 16,384 tokens (164M tokens), 2% warmup, cosine to lr/100, Adam
  beta2 0.999; lr 2e-3 with the overlay, 4e-3 without (fixed by `sd12lr`, seed 0).
- Evaluation: the same 32 held-out 16k windows for every run (eval seed 987654); per-token NLL on the
  training host (`post_eval`); mean loss = mean NLL over scored tokens.

## Part A. Dose grid (exploratory by design: 2 seeds)
Grid: A in {0, 1, 2, 4} attention layers x P in {0, 1, 2, 4} placements, seeds 0-1 (`sd14-*`),
plus pure GDN without the overlay at lr 8e-3.

**A1. Iso-loss exchange rate.** Reference loss L_ref = mean over seeds of the A = 2, P = 0 cell (the
3:1 hybrid without the overlay, today's frontier shape). For each P >= 1, the attention needed to match
it, A*(P), is found by linear interpolation of the seed-mean loss over A in {0, 1, 2, 4} at that P.
If every A at that P is below L_ref, report A*(P) <= 0 ("the overlay alone matches the 3:1 hybrid"),
with the margin L_ref - L(A = 0, P). Reported quantity: attention layers saved, 2 - A*(P), per P.
"More placements, less attention" is supported if A*(P) is non-increasing in P on both seeds'
interpolations; otherwise we report the shape as observed.

**A2. Loss surface.** All 16 cells: seed means, both seeds, and per-type losses (A3). No smoothing or
model fit beyond A1's interpolation.

**A3. Per-dependency-type cost of removing attention.** Types fixed now (src/tagger.py v1):
exact_far, exact_near, consistent, latest_value, older_value, fuzzy_far, no_match, novel, format,
count_next. For each type and each P, the cost of removing attention is
C(type, P) = mean NLL(A = 0, P) - mean NLL(A = 2, P), with a 95% bootstrap CI over the 32 eval windows
(10k resamples, windows resampled jointly across arms, seeds averaged). Expectations, written now:
exact_far cost at P >= 1 below its cost at P = 0; fuzzy_far and no_match costs change less with P.

**A4. Attention hand-off.** In the A = 2 and A = 4 cells, the induction share of far attention
(far_ind / far, averaged over heads and layers) at P = 0 versus P = 1, 2, 4. Expectation: lower with
the overlay. Reported with both seeds; no test.

## Part B. Confirmatory test 1: attention-free + overlay vs the 3:1 hybrid
- Arms: A = 0, P = 1 (GDN x 8 + additive, lr 2e-3) versus A = 2, P = 0 (3:1 hybrid without the
  overlay, lr 4e-3). Configurations as in Part A.
- Seeds 20-27 (n = 8 per arm), never used before.
- Primary metric: mean held-out loss at step 10,000. Test: one-sided Wilcoxon signed-rank on paired
  differences (A0P1 - A2P0 < 0), alpha 0.05, single test. Also reported: mean difference with a 95%
  bootstrap CI over seeds (10k resamples) and the paired t-test.
- Reading: p < 0.05 gives "the attention-free model with the write path reaches lower held-out loss
  than the 3:1 hybrid by <mean> [CI] at 51M parameters, 16k context"; otherwise "no detectable
  difference at n = 8". Failed runs are relaunched with the same seed; no seed is dropped.

## Part C. Confirmatory test 2: locker_a vs additive
- Arms: ctxkey with K = 4 notes (`L5` + `overlay.ctx_key`, `overlay.notes` 4), lr 2e-3, versus
  additive (`ADD`), lr 2e-3 (each arm's best LR from `sd11`).
- Setting: 4 blocks d_model 256, 4k context, batch 4, 10,000 steps, SD_TASK (4k pool), tuned recipe.
- Seeds 13-20 (n = 8 per arm), never used for these arms.
- Same test and reporting as Part B, direction ctxkey - additive < 0.
- Secondary (reported, not tested): K = 8 on the same seeds, for the notes dial.
- Stated in advance: ctxkey K = 4 holds 8 vectors of slot state per cue against additive's 1, and
  runs ~1.4x slower per step in this implementation (0.145 vs 0.10 s); the test is matched in steps,
  data and LR tuning, not in state or wall-clock.
