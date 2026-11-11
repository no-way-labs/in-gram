# Pre-registration files

The paper states that the arms, the metric and the test of each confirmatory comparison were committed before
the runs of its new arms started. These are those files, reproduced without changes from the private
development repository, with the time of each commit and the time at which the first run of each phase was
launched. All times are UTC.

| File | Commit | Committed | First run of the phase launched |
|---|---|---|---|
| `PREREG_PHASE27.md` | `46bff155` | 2026-11-09 00:51:25 | 2026-11-09 05:45:30 |
| `PREREG_PHASE29_first_commit.md` | `1391b024` | 2026-11-09 16:02:35 | 2026-11-09 17:04:45 |
| `PREREG_PHASE29.md` (adds the addendum on the grid rerun) | `0fe3c69b` | 2026-11-09 16:19:19 | 2026-11-09 17:04:45 |
| `PREREG_PHASE30.md` | `ea5c014b` | 2026-11-09 18:31:24 | 2026-11-09 18:31:42 |
| `PREREG_PHASE31.md` | `79cbf5d4` | 2026-11-09 20:05:31 | 2026-11-09 20:31:52 |
| `PREREG_PHASE32.md` | `b36d2bb8` | 2026-11-10 00:52:44 | 2026-11-10 00:53:05 |

The launch times come from the billing records of the compute account, which list the 200 most recent
campaigns; the time given for Phase 27 is the earliest campaign in that list. The header of `PREREG_PHASE31.md` gives
20:10Z as the time of writing, five minutes after its commit; the file is reproduced as committed.

## Where each test appears in the paper

| Pre-registered test | In the paper |
|---|---|
| Phase 27, Part A (dose grid, exploratory) | Section 4.4, Figures 5 and 6, and the appendix |
| Phase 27, Part B (cue lengths 1 to 5 against the 3:1 hybrid) | Appendix, "Per-seed losses" |
| Phase 27, Part C (scored read against the additive read) | Appendix, "Read rule, quantization and generation" |
| Phase 29, primary (long cues against cue lengths 1 to 5) | Appendix, "Cue lengths and further checks" |
| Phase 29, secondary 1 and 2 | Figure 3, row 1, without and with the n-gram cache |
| Phase 30, primary and secondary | Figure 3, row 2, without and with the n-gram cache |
| Phase 31, T3 | Figure 3, row 3 (static N-gram table) |
| Phase 31, T4 | Figure 3, row 4 (12 blocks of width 384) |
| Phase 31, T5 | Figure 3, row 5 (16 blocks of width 512, four seeds, reported without a test) |
| Phase 31, T1 | Figure 3, row 6 (both models at 328M training tokens) |
| Phase 31, T2 | Figure 3, row 7 (the hybrid trained twice as long) |
| Phase 32, T1 and T2 | Figure 3, rows 8 and 9 (1.31B training tokens) |

## Names used in these files

The files were written during development and use its working names.

| In these files | In the paper |
|---|---|
| locker, overlay, write path, `ovl` | In-gram |
| additive, `ADD`, `LOCKER31` | the read `R + S` and the configuration of Section 2.4 |
| ctxkey, `locker_a`, K notes | the scored read of the appendix |
| post-eval NLL | loss |
| dev seeds 0-1 | the selection seeds |
| `conf6`, `conf29`, `conf30`, `b31-*`, `p32-*`, `sd14`, `sd29` | identifiers of groups of runs |
| free n-gram cache | the n-gram cache baseline |

Scripts and documents that the files mention by path (`scripts/...`, `docs/...`, `DEVIATIONS.md`) belong to the
development repository and are not part of this release.
