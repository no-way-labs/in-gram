# Reproducing the paper

This folder holds everything behind the paper's numbers: the code that ran, the configuration of every run, the
log and the per-token losses of every run, and the scripts that turn them into the tables and figures. There
are three levels of reproduction, from a check that takes seconds to retraining the models.

The code here is the research code as it ran, and it uses the working names of the project: `locker`, `overlay`
and the arm `ovl` all mean In-gram, and `base` is a model without it. The table in
[`../preregistration/README.md`](../preregistration/README.md) maps the other names. For a short, readable
implementation of the layer alone, see [`../ingram`](../ingram); it was checked numerically against
`src/overlay.py`.

## Setup

```bash
cd research
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt     # on a machine without CUDA, leave out flash-linear-attention
pytest tests                        # 280 tests pass on a CPU; 21 need a GPU and are skipped there
```

## Level 1: regenerate every number from the logs (no GPU, no data, under a minute)

```bash
python scripts/paper_results.py          # tables, figure data and paper/generated/numbers.txt
python scripts/serving_cost.py --tex     # the two serving scenarios
git status --short ../paper/generated    # prints nothing: the files are reproduced byte for byte
cd ../paper && latexmk -pdf -outdir=build main.tex
```

`runs/` holds, for each of the 199 training runs that the paper's numbers depend on, the training log
(`.jsonl`, whose first line is the full configuration), the per-token losses on the 32 held-out windows
(`.npz`, an array of 32 by 16,384 values) and, where the paper uses them, the attention statistics
(`.post_eval.json`). `runs/tags/` holds the token types and the n-gram cache probabilities of those windows,
and `runs/q28/` the quantization and generation checks of the appendix. Every number in the text of the paper is
printed with its source in `paper/generated/numbers.txt`.

## Level 2: recompute the data statistics (CPU, 3.2 GB of disk)

These scripts read the tokenized dataset, which is built on first use into `$TEXT_CACHE` (default `./data`).
The build streams the first 16,384 trajectories of
[`nebius/SWE-rebench-openhands-trajectories`](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
from Hugging Face, renders and tokenizes them, and splits them by repository (15,680 training and 704 held-out
trajectories).

```bash
python scripts/gram_stats_by_context.py 704     # repeat statistics by position in the session (Figure 2)
python scripts/gram_stats_by_context.py 704 131072
python scripts/slots_from_start.py              # slots per token of context (Figure 7)
python scripts/slot_growth.py                   # the same count on the evaluation windows
python scripts/figure1_example.py               # the counts behind the example of Figure 1
python verify_data.py                           # compares your token files with data_manifest.json
python scripts/tag_eval_set.py 16384 32 1 eval16k   # token types of the evaluation windows
```

Each script writes a file in `results/` (or `runs/tags/`) that already exists here, and a rerun can therefore be
compared with `git diff`. `verify_data.py` checks the SHA-256 of your token files against the ones of the dataset as
downloaded on 2026-11-08; if the dataset on Hugging Face has changed since, the hashes differ and losses on the
new data are not comparable token for token.

## Level 3: retrain a run (one GPU)

```bash
python run.py configs/runs/ovl-conf29-o12481632-s20.json --smoke    # 4 steps on a CPU, to check the setup
python run.py configs/runs/ovl-conf29-o12481632-s20.json --out out  # the real run
```

`configs/runs/` has one file per run with the configuration that the run logged, and
`configs/index.csv` lists every run with its group, seed, steps, learning rate, parameter count, the loss it
reached and its training time. `run.py` trains with `src/train.py`, then evaluates the final model on the 32
held-out windows with `src/nll_dump.py` and prints the loss beside the paper's.

| Runs | Group in `configs/index.csv` | Steps | Minutes per run on an RTX 4090 |
|---|---|---|---|
| GDN + In-gram, 8 blocks of width 256 | `conf29-o12481632` | 10,000 | 19 to 33 |
| 3:1 hybrid, 8 blocks of width 256 | `conf6-a2p0` | 10,000 | 17 to 21 |
| 3:1 hybrid, 12 blocks of width 384 (parameter-matched) | `conf30-hyb54` | 10,000 | 25 to 29 |
| 3:1 hybrid with a static N-gram table | `b31-static` | 10,000 | 21 to 25 |
| GDN + In-gram, 12 blocks of width 384 | `b31-size2` | 10,000 | 27 to 31 |
| 16 blocks of width 512, both models | `b31-size3-locker`, `b31-size3-hybrid` | 20,000 | 67 to 74 |
| Both models at 328M training tokens | `b31-longloc`, `b31-longhyb` | 20,000 | 39 to 55 |
| 7:1 hybrid + In-gram, 3:1 hybrid, 7:1 hybrid | `p32-a1p1`, `p32-a2p0`, `p32-a1p0` | 80,000 | 124 to 213 |
| Dose grid | `sd14-*` (without In-gram), `sd29-*` (with) | 10,000 | 17 to 36 |
| Learning-rate and cue-length selection | `a31-*`, `sd15-*`, `sd14x-*`, `cue28-*`, `cue29-*` | 10,000 or 20,000 | see the index |
| Scored read (appendix) | `conf7-*` | 10,000 | see the index |

All 199 runs together took 148 hours of training on rented RTX 4090s. The runs used Python 3.12, PyTorch 2.14
with bfloat16 autocast, and `flash-linear-attention` 0.6.0 for Gated DeltaNet.

What to expect from a rerun:

- The data order and the initialization are fixed by the seed, but the GPU kernels are not deterministic bit
  for bit, and we have not rerun the models on other hardware. A rerun should therefore land near the loss in
  `configs/index.csv`, not on it; the spread across the eight seeds of a group, which the index shows, is the
  scale to compare with.
- To feed a retrained run into the analysis, write it to `runs/<group>-s<seed>/`, the directory its original
  occupies, and rerun Level 1.

The other experiments of the appendix have their own entry points: `src/quant_eval.py` (quantization and
generation, on the checkpoints of the runs `sd14-a0p0`, `sd14-a2p0`, `sd14-a0p1` and `sd14-a0p2`, seeds 0 and
1), `scripts/bench_decode.py` (the decode benchmark) and `scripts/analyze_phase27.py` to
`scripts/analyze_phase32.py` (the analyses named in the pre-registration files).

## What is not here

- **Checkpoints.** Each is 0.12 to 0.46 GB, and the 199 of them are not in the repository.
- **Launch scripts.** The runs were launched on rented machines through an orchestration service, and those
  scripts depend on its private SDK. `run.py` runs the same training and evaluation code on one machine.
- **Code for baselines that the paper does not report.** A few model variants in `src/models.py` import
  third-party code that is not redistributed here; none of the paper's runs uses them.
