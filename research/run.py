"""Train one run of the paper on this machine, then evaluate it as the paper does.

    python run.py configs/runs/ovl-conf29-o12481632-s20.json            # the full run (one GPU with CUDA)
    python run.py configs/runs/ovl-conf29-o12481632-s20.json --smoke    # a few steps on the CPU, to check the setup

Each file in configs/runs/ is the configuration of one run that the paper's numbers depend on. A run trains with
src/train.py, writes its log and checkpoint to --out, and is then evaluated on the 32 held-out windows of 16,384
tokens (src/nll_dump.py), which gives the per-token losses that the analysis scripts read. The loss that the
paper's run reached is in configs/index.csv.

The first run downloads and tokenizes the SWE-rebench trajectories into $TEXT_CACHE (default: ./data).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TEXT_CACHE", str(HERE / "data"))
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="a file in configs/runs/")
    ap.add_argument("--out", default="out", help="directory for the log, the checkpoint and the per-token losses")
    ap.add_argument("--smoke", action="store_true",
                    help="4 steps on 512-token windows of the first 400 trajectories, with attention in place of GDN "
                         "so that no GPU is needed")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    run, post = cfg["run"], cfg.get("post_eval")
    if args.smoke:
        ov = dict(run.get("overrides") or {})
        ov.update(mixer=["attn" if m == "gdn" else m for m in ov.get("mixer", [])], rope=True, pos_emb=False, attn_conv=4)
        task = dict(run["task"], L_max=512, max_rows=400)  # the first 400 trajectories are enough for a check
        run = dict(run, run_id=run["run_id"] + "-smoke", steps=4, eval_every=2, eval_size=2, warmup=1, task=task,
                   overrides=ov, device="cpu")
        post = dict(post, n=2, task=dict(post["task"], L_max=512, max_rows=400)) if post else None

    from src.train import run as train_run

    out = Path(args.out)
    train_run(run, out)
    print("training finished; log and checkpoint are in", out)
    if post:
        from src.nll_dump import dump

        res = dump({**post, "out": str(out), "ckpts": [str(out / f"{run['run_id']}.pt")]})
        mean = res["models"].get(run["run_id"], {}).get("mean_nll")
        print(f"held-out loss of this run: {mean:.4f} nats per token" if mean is not None else res)
        if not args.smoke:
            with open(HERE / "configs" / "index.csv") as f:
                paper = {r["run_id"]: r["loss"] for r in csv.DictReader(f)}
            print(f"loss of the same run in the paper: {paper.get(run['run_id'], 'not listed')}")


if __name__ == "__main__":
    main()
