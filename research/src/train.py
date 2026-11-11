"""Training loop, eval, logging (plan.md §7) + diagnostics.

run(config, out_dir) is idempotent: a run whose JSONL ends with a "final"
record is skipped; otherwise it resumes from <run_id>.pt if present.

Config keys (defaults in DEFAULTS): run_id, arm, task (dict), overrides (dict),
steps, batch, lr, warmup, wd, clip, eval_every, eval_size, eval_seed, seed,
device, num_workers.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from .controls import arm_config
from .models import Model, block_mixers, n_params
from .schedule import build_schedule
from .tasks import IGNORE, PT_NAMES, PT_QFINAL, make_task, round_up

DEFAULTS = dict(
    steps=6000, batch=64, lr=1e-3, warmup=100, wd=0.1, clip=1.0, eval_every=200,
    eval_size=1024, eval_seed=987654, seed=0, device=None, num_workers=None, overrides=None,
)

INT_KEYS = ("ev_flat", "ev_order", "ev_b", "ev_pos", "ev_h", "ev_seg_start", "read_ptr", "read_row", "j_read", "j_write")


def sched_to_torch(s: dict, device) -> dict:
    out = {}
    for k, v in s.items():
        if k in INT_KEYS:
            out[k] = torch.as_tensor(v, dtype=torch.long, device=device)
        elif k in ("max_seg_len", "j_slots"):
            out[k] = int(v)
    return out


def with_lmax(task: dict) -> dict:
    """Fill L_max for KV tasks: C + Q*(k+1) + 1 rounded up to a multiple of 64 (plan 1a)."""
    task = dict(task)
    if task.get("name") == "lm":
        task.setdefault("L_max", 1024)
        return task
    if task.get("name") == "agentic":
        task.setdefault("L_max", 4096)
        return task
    if task.get("name") == "topic_kv" and "L_max" not in task:
        k, T, Ct, Q = task.get("k", 2), task.get("T", 2), task.get("C_t", 192), task.get("Q", 32)
        task["L_max"] = round_up(1 + T * (1 + Ct) + Q * (k + 2), 64)
        return task
    if task.get("name", "kv") == "kv" and "L_max" not in task:
        k, C, Q = task.get("k", 2), task.get("C", 384), task.get("Q", 32)
        task["L_max"] = round_up(C + Q * (k + 1) + 1, 64)
    return task


def cpu_threads() -> int:
    return int(os.environ.get("RHOMBIC_CPU_THREADS") or os.cpu_count() or 1)


def make_batch(batch_fn, sched_cfg, B, seed):
    rng = np.random.default_rng(seed)
    batch = batch_fn(B, rng)
    # Canonicalized ids (if the task provides them) drive hashing only; the model sees raw tokens.
    batch["sched"] = build_schedule(batch.get("hash_tokens", batch["tokens"]), seed=seed, **sched_cfg)
    return batch


class StepStream(IterableDataset):
    """Yields (step, batch) for steps [start, steps). Each worker takes every
    n-th step, so the DataLoader round-robin restores step order. Holds only
    plain data (task dict, not a closure) so spawn-started workers can pickle it."""

    def __init__(self, task: dict, sched_cfg, B, seed, start, steps):
        self.task, self.sched_cfg, self.B = dict(task), dict(sched_cfg), B
        self.seed, self.start, self.steps = seed, start, steps

    def __iter__(self):
        _, batch_fn = make_task(self.task)
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        for step in range(self.start + wid, self.steps, nw):
            yield step, make_batch(batch_fn, self.sched_cfg, self.B, self.seed * 1000 + step)


def _resolve_matched_arms(arm, mcfg, scfg, task, eval_batches):
    """base_param / base_state: match params (2%) or state bytes (10%) of `ovl`."""
    if mcfg.get("mlp_hidden") is not None and mcfg.get("heads") is not None:
        return mcfg
    ovl_m, ovl_s = arm_config("ovl", task)
    if mcfg.get("mlp_hidden") is None:  # base_param
        # MLP params per block = h*(2d+1) + d, so the extra width is solved directly.
        ovl_extra = n_params(Model(dict(ovl_m, mixer="none"))) - n_params(Model(dict(ovl_m, mixer="none", overlay=None)))
        d, nb = mcfg["d_model"], mcfg["n_blocks"]
        mcfg["mlp_hidden"] = ovl_m["mlp_hidden"] + round(ovl_extra / (nb * (2 * d + 1)))
    if mcfg.get("heads") is None:  # base_state
        mean_slots = np.mean([b["sched_ovl"]["n_slots"].mean() for b in eval_batches]) if eval_batches else 0
        ovl_state = mean_slots * ovl_m["overlay"]["d_mem"] * 4
        gdn_state = lambda h: mcfg["n_blocks"] * h * 64 * 64 * 4
        mcfg["heads"] = max(2, round((gdn_state(2) + ovl_state) / gdn_state(1)))
    return mcfg


def state_bytes(mcfg, scfg, mean_slots, L):
    d, nb = mcfg["d_model"], mcfg["n_blocks"]
    s = 0
    for m in block_mixers(mcfg):
        if m == "attn":
            s += 2 * L * d * 4
        elif m in ("gdn", "gla"):
            s += mcfg["heads"] * 64 * 64 * 4
        elif m == "hgdn":  # dense state + one 64-float column per head per written cue slot
            s += mcfg["heads"] * 64 * 64 * 4 + (mean_slots * mcfg["heads"] * 64 * 4 if mcfg["hash"]["orders"] else 0)
        elif m == "sdm":  # slot bank: heads * slots * head_dim floats
            s += mcfg["heads"] * mcfg.get("sdm_slots", 1024) * (d // mcfg["heads"]) * 4
    ov = mcfg.get("overlay")
    if ov and ov.get("kind", "memory") == "memory":
        if scfg["mode"] == "growing":
            per = mean_slots * ov["d_mem"] * 4
        else:
            per = len(scfg["orders"]) * scfg["H"] * scfg["M"] * (ov["d_mem"] * 4 + 4)
        s += per * len(Model.locker_points(mcfg))
    return float(s)


def param_groups(model, wd, table_lr_mult=1.0):
    """table_lr_mult: learning-rate multiplier for the overlay's learned prior table (prior_table)."""
    decay, no_decay, table = [], [], []
    for name, p in model.named_parameters():
        if table_lr_mult != 1.0 and name.endswith("prior_table"):
            table.append(p)
        elif p.ndim < 2 or "embed" in name or "pos." in name or "norm" in name or name.endswith("table"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [dict(params=decay, weight_decay=wd), dict(params=no_decay, weight_decay=0.0)]
    if table:
        groups.append(dict(params=table, weight_decay=0.0, lr_mult=table_lr_mult))
    return groups


def grad_norms(model) -> dict:
    groups = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        key = name.split(".")[0]
        if key == "blocks":
            key = "mixer" if ".mixer." in name else "mlp"
        if key == "overlays":
            parts = name.split(".")
            key = f"locker@{parts[1]}.{parts[2]}"
        groups[key] = groups.get(key, 0.0) + float(p.grad.detach().float().pow(2).sum())
    return {k: math.sqrt(v) for k, v in groups.items()}


def read_correct_rate(batch, scfg, k):
    """Schedule-level probe: at clean query-final positions, does the read
    pointer (order k, each head) land on the write from the answer's value token?"""
    if k not in scfg["orders"] or scfg["read"] != "normal":
        return None
    oi = list(scfg["orders"]).index(k)
    s = batch["sched"]
    b, t = np.nonzero((batch["pos_type"] == PT_QFINAL) & ~batch["corrupt"])
    hits = []
    for h in range(scfg["H"]):
        p = s["read_ptr"][b, t, oi, h]
        ok = (p >= 0) & (s["ev_pos"][np.maximum(p, 0)] == batch["src_pos"][b, t])
        hits.append(ok)
    return float(np.mean(hits)) if len(b) else None


@torch.no_grad()
def evaluate(model, eval_batches, device, ablate=False, detail=True):
    """ablate: False | "all" (y = 0 everywhere) | "query" (y = 0 only at scored positions)
    | "miss" (reads at scored positions replaced by empty reads: "as if the cache missed")."""
    model.eval()
    lema = lema_layers(model)
    for m in lema:  # LEMA evaluates with the exact op (their load_trained sets hard mode)
        m.hard = not getattr(m, "_force_soft", False)
    acc = dict(n=0, correct=0, loss=0.0, n_c=0, correct_c=0, n_k=0, correct_k=0)
    pt_alpha = np.zeros(len(PT_NAMES))
    pt_alpha_n = np.zeros(len(PT_NAMES))
    pt_beta = np.zeros(len(PT_NAMES))
    pt_beta_n = np.zeros(len(PT_NAMES))
    pt_y = np.zeros(len(PT_NAMES))
    pt_loss = np.zeros(len(PT_NAMES))
    pt_correct = np.zeros(len(PT_NAMES))
    pt_n = np.zeros(len(PT_NAMES))
    for batch in eval_batches:
        tokens = torch.as_tensor(batch["tokens"], device=device)
        targets = torch.as_tensor(batch["targets"], device=device)
        sched = sched_to_torch(batch["sched"], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ab = {False: False, "all": True}.get(ablate, None)
            if ablate == "query":  # locker's output removed at the answer position only
                ab = targets != IGNORE
            elif ablate == "miss":  # the answering read becomes an empty read
                ab = {"miss": targets != IGNORE}
            h, stats = model(tokens, sched, ablate=ab, return_hidden=True)
        mask = targets != IGNORE
        tg = targets[mask]
        preds, nlls = [], []
        off = 0
        for lgc in head_chunks(model, h[mask], device):
            tc = tg[off:off + lgc.shape[0]]
            preds.append(lgc.argmax(-1))
            nlls.append(F.cross_entropy(lgc, tc, reduction="none"))
            off += lgc.shape[0]
        pred = torch.cat(preds) if preds else tg
        nll_all = torch.cat(nlls) if nlls else tg.float()
        cor = torch.as_tensor(batch["corrupt"], device=device)[mask]
        acc["n"] += int(mask.sum())
        acc["correct"] += int((pred == tg).sum())
        acc["loss"] += float(nll_all.sum())
        acc["n_c"] += int(cor.sum())
        ptm = torch.as_tensor(batch["pos_type"], device=device)[mask].long().cpu().numpy()
        nll = nll_all.cpu().numpy()
        np.add.at(pt_loss, ptm, nll)
        np.add.at(pt_correct, ptm, (pred == tg).cpu().numpy())
        np.add.at(pt_n, ptm, 1)
        acc["correct_c"] += int(((pred == tg) & cor).sum())
        if detail and stats:
            pt = batch["pos_type"]
            al = stats["alpha"].cpu().numpy()
            yn = stats["y_norm"].cpu().numpy()
            np.add.at(pt_alpha, pt.ravel(), al.ravel())
            np.add.at(pt_y, pt.ravel(), yn.ravel())
            np.add.at(pt_alpha_n, pt.ravel(), 1)
            if "beta" in stats:
                ev_pt = pt.ravel()[batch["sched"]["ev_flat"]]
                np.add.at(pt_beta, ev_pt, stats["beta"].cpu().numpy())
                np.add.at(pt_beta_n, ev_pt, 1)
    model.train()
    for m in lema:
        m.hard = False
    n_clean = acc["n"] - acc["n_c"]
    out = dict(
        acc=acc["correct"] / max(acc["n"], 1),
        loss=acc["loss"] / max(acc["n"], 1),
        acc_clean=(acc["correct"] - acc["correct_c"]) / n_clean if n_clean else None,
        acc_corrupt=acc["correct_c"] / acc["n_c"] if acc["n_c"] else None,
    )
    if detail and (pt_n > 0).sum() > 1:  # loss/acc per scored-position type (e.g. text_rep vs text_novel)
        out["loss_by_type"] = {PT_NAMES[i]: round(float(pt_loss[i] / pt_n[i]), 5) for i in range(len(PT_NAMES)) if pt_n[i]}
        out["acc_by_type"] = {PT_NAMES[i]: round(float(pt_correct[i] / pt_n[i]), 5) for i in range(len(PT_NAMES)) if pt_n[i]}
        out["frac_by_type"] = {PT_NAMES[i]: round(float(pt_n[i] / pt_n.sum()), 5) for i in range(len(PT_NAMES)) if pt_n[i]}
    if detail and pt_alpha_n.sum():
        nz = lambda s, n: {PT_NAMES[i]: round(float(s[i] / n[i]), 5) for i in range(len(PT_NAMES)) if n[i]}
        out["alpha"] = nz(pt_alpha, pt_alpha_n)
        out["y_norm"] = nz(pt_y, pt_alpha_n)
        if pt_beta_n.sum():
            out["beta"] = nz(pt_beta, pt_beta_n)
    return out


CE_CHUNK = 4096  # scored positions per output-head chunk


def chunked_ce(model, h, targets, device):
    """Mean cross-entropy over scored positions, applying the output head in checkpointed chunks:
    only one chunk of [rows, vocab] logits is alive at a time (50k-vocab LM fits on one GPU)."""
    from torch.utils.checkpoint import checkpoint

    mask = targets != IGNORE
    hs, ts = h[mask], targets[mask]
    n = ts.numel()
    total = h.new_zeros((), dtype=torch.float32)

    def piece(hc, tc):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model.head(hc)
        return F.cross_entropy(lg.float(), tc, reduction="sum")

    for i in range(0, n, CE_CHUNK):
        total = total + checkpoint(piece, hs[i:i + CE_CHUNK], ts[i:i + CE_CHUNK], use_reentrant=False)
    return total / max(n, 1)


def head_chunks(model, hs, device):
    """Yield logits [rows, vocab] (fp32) for scored hidden states, chunk by chunk (eval)."""
    for i in range(0, hs.shape[0], CE_CHUNK):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model.head(hs[i:i + CE_CHUNK])
        yield lg.float()


def lema_layers(model):
    from .models import LemaMixer

    return [m for m in model.modules() if isinstance(m, LemaMixer)]


def set_lema_hardening(model, step, steps):
    """LEMA's LM recipe (their experiments/main_lm/train_lm.py), as fractions of the run: c ramps
    linearly 0 -> d_qk - 1 over [2%, 10%]; alpha = 1/sqrt(d_qk) until 10%, then linear to 10 at the
    end; backward cap 2; beta 4 (set at construction)."""
    for m in lema_layers(model):
        u = min(max((step - 0.02 * steps) / (0.08 * steps), 0.0), 1.0)
        m.c = (m.dh - 1) * u
        a0 = m.dh ** -0.5
        m.alpha = a0 if step <= 0.1 * steps else min(10.0, a0 + (10.0 - a0) * (step - 0.1 * steps) / (0.9 * steps))
        m.cap = 2.0


def _lr_at(step, lr, warmup, schedule="constant", steps=1, lr_final_frac=0.01):
    """Linear warmup, then constant (default) or a whole-run cosine to lr * lr_final_frac
    (LEMA's LM recipe uses the cosine)."""
    warm = min(1.0, (step + 1) / warmup)
    if schedule == "constant":
        return lr * warm
    t = min(step / max(steps - 1, 1), 1.0)
    return warm * lr * (lr_final_frac + (1 - lr_final_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def run(config: dict, out_dir: str | Path) -> dict:
    cfg = {**DEFAULTS, **config}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = cfg["run_id"]
    log_path, ckpt_path = out_dir / f"{run_id}.jsonl", out_dir / f"{run_id}.pt"

    if log_path.exists():
        lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        if lines and lines[-1].get("final"):
            return lines[-1]

    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    threads = cpu_threads()
    torch.set_num_threads(max(1, threads if device.type == "cpu" else min(threads, 4)))
    torch.manual_seed(cfg["seed"])

    task = with_lmax(cfg["task"])
    task_cfg, batch_fn = make_task(task)
    # Extra eval tasks (e.g. longer contexts for length generalization), run at the end.
    extra_tasks = {name: with_lmax({**{k_: v for k_, v in task.items() if k_ != "L_max"}, **over})
                   for name, over in (cfg.get("extra_evals") or {}).items()}
    overrides = dict(cfg["overrides"] or {})
    if task.get("name") == "lm":
        from .text import vocab_size

        overrides.setdefault("vocab", vocab_size())
    elif task.get("name") == "agentic":
        from .agentic import vocab_size as agentic_vocab

        overrides.setdefault("vocab", agentic_vocab(task.get("tokenizer", "gpt2")))
    # Position table must cover every eval length; positions never seen in training stay untrained.
    overrides.setdefault("max_len", max([task["L_max"]] + [t["L_max"] for t in extra_tasks.values()]))
    mcfg, scfg = arm_config(cfg["arm"], task, overrides)
    k = task.get("k", 2)

    # Fixed eval set: identical across arms of a comparison (same eval_seed). Real text evaluates on
    # the validation split.
    if task.get("name") in ("lm", "agentic"):  # agentic: 'validation' = held-out repos
        _, eval_fn = make_task({**task, "split": "validation"})
    else:
        eval_fn = batch_fn
    eval_batches = []
    ers = np.random.default_rng(cfg["eval_seed"])
    for i in range(0, cfg["eval_size"], cfg["batch"]):
        b = eval_fn(min(cfg["batch"], cfg["eval_size"] - i), ers)
        b["sched"] = build_schedule(b.get("hash_tokens", b["tokens"]), seed=cfg["eval_seed"] + i, **scfg)
        if mcfg.get("heads") is None:
            _, s_ovl = arm_config("ovl", task)
            b["sched_ovl"] = build_schedule(b["tokens"], **s_ovl)
        eval_batches.append(b)
    mcfg = _resolve_matched_arms(cfg["arm"], mcfg, scfg, task, eval_batches)

    mean_slots = float(np.mean([b["sched"]["n_slots"].mean() for b in eval_batches]))
    rc = [read_correct_rate(b, scfg, k) for b in eval_batches]
    rc = float(np.mean(rc)) if rc and rc[0] is not None else None

    model = Model(mcfg).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg["wd"], cfg.get("table_lr_mult", 1.0)), lr=cfg["lr"],
                            betas=(0.9, cfg.get("adam_beta2", 0.95)))
    start = 0
    elapsed = 0.0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, elapsed = ck["step"], ck["elapsed"]
        kept = [l for l in log_path.read_text().splitlines() if l.strip()
                and (json.loads(l).get("step", -1) <= start or "config" in json.loads(l))]
        log_path.write_text("\n".join(kept) + "\n")
    else:
        header = dict(config=cfg, task=task, model=mcfg, sched={k_: (list(v) if isinstance(v, tuple) else v)
                                                                  for k_, v in scfg.items()},
                      n_params=n_params(model), state_bytes=state_bytes(mcfg, scfg, mean_slots, task_cfg.L_max),
                      mean_slots=mean_slots, read_correct=rc, device=str(device), threads=threads)
        log_path.write_text(json.dumps(header) + "\n")

    def log(rec):
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    nw = cfg["num_workers"]
    if nw is None:
        nw = min(8, max(2, threads // 3)) if device.type == "cuda" else min(4, max(0, threads // 8))
    loader = DataLoader(StepStream(task, scfg, cfg["batch"], cfg["seed"], start, cfg["steps"]),
                        batch_size=None, num_workers=nw, persistent_workers=False,
                        prefetch_factor=4 if nw else None,
                        multiprocessing_context=cfg.get("mp_context") if nw else None)
    has_overlay = model.overlay is not None
    has_lema = bool(lema_layers(model))
    t0 = time.time() - elapsed
    t_data = 0.0
    t_last = time.time()
    escape = None
    model.train()
    prof_win = cfg.get("profile_steps")  # [start, stop): torch.profiler table logged at stop
    prof = None
    step_times, t_step = [], time.time()
    for step, batch in loader:
        t_data += time.time() - t_last
        if prof_win and step == prof_win[0]:
            from torch.profiler import ProfilerActivity, profile

            acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
            prof = profile(activities=acts)
            prof.start()
        if prof is not None and step == prof_win[1]:
            if device.type == "cuda":
                torch.cuda.synchronize()
            prof.stop()
            ka = prof.key_averages()
            key = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
            log(dict(profile=ka.table(sort_by=key, row_limit=40),
                     profile_cpu=ka.table(sort_by="self_cpu_time_total", row_limit=25), steps=prof_win))
            prof = None
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg["lr"], cfg["warmup"], cfg.get("lr_schedule", "constant"),
                             cfg["steps"]) * g.get("lr_mult", 1.0)
        if has_lema:
            set_lema_hardening(model, step, cfg["steps"])
        tokens = torch.as_tensor(batch["tokens"], device=device)
        targets = torch.as_tensor(batch["targets"], device=device)
        sched = sched_to_torch(batch["sched"], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h, _ = model(tokens, sched, return_hidden=True)
        loss = chunked_ce(model, h, targets, device)
        aux = [m.aux_loss for m in model.modules() if getattr(m, "aux_loss", None) is not None]
        if aux:  # DSA indexer KL (its input is detached, so this trains only the indexer)
            loss = loss + sum(aux)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        is_eval = (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps"]
        gn = grad_norms(model) if is_eval else None
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        opt.step()
        if cfg.get("time_steps"):  # synced per-step wall time + batch shape (speed diagnosis)
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            step_times.append((step, round(now - t_step, 4), int(batch["sched"]["ev_flat"].shape[0]),
                               int(np.asarray(batch["sched"]["max_seg_len"]))))
            t_step = now
            if (step + 1) % 100 == 0:
                log(dict(step_times=step_times))
                step_times = []
        if is_eval:
            if device.type == "cuda":
                torch.cuda.synchronize()
            wall = time.time() - t0
            ev = evaluate(model, eval_batches, device)
            rec = dict(step=step + 1, train_loss=float(loss.detach()), **ev, grad_norm=gn, wall=wall,
                       gpu_seconds=wall if device.type == "cuda" else 0.0, data_wait=t_data,
                       sec_per_step=wall / (step + 1 - start) if step + 1 > start else None)
            if device.type == "cuda":  # peak memory since the previous eval (efficiency study)
                rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
                torch.cuda.reset_peak_memory_stats()
            if has_overlay:
                for key, mode in (("ablate", "all"), ("ablate_q", "query"), ("miss_q", "miss")):
                    ab = evaluate(model, eval_batches, device, ablate=mode, detail=False)
                    rec[f"acc_{key}"] = ab["acc"]
                    if ab["acc_corrupt"] is not None:  # split, so clean and fuzzy paths can be told apart
                        rec[f"acc_{key}_clean"], rec[f"acc_{key}_corrupt"] = ab["acc_clean"], ab["acc_corrupt"]
            if escape is None and ev["acc"] >= 0.8:
                escape = step + 1
            log(rec)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1, elapsed=wall,
                            mcfg=mcfg, scfg=scfg),
                       str(ckpt_path) + ".tmp")
            os.replace(str(ckpt_path) + ".tmp", ckpt_path)
        t_last = time.time()

    extra = {}
    if has_lema:  # final eval with the surrogate as trained (vs the exact op used by evaluate())
        for m in lema_layers(model):
            m._force_soft = True
        ev_soft = evaluate(model, eval_batches, device, detail=True)
        for m in lema_layers(model):
            m._force_soft = False
        extra["lema_soft_loss"] = ev_soft["loss"]
        log(dict(lema_soft_eval=True, loss=ev_soft["loss"], loss_by_type=ev_soft.get("loss_by_type")))
    modes = cfg.get("final_read_modes") or []
    if modes and has_overlay:  # locker5 inference ablations on the final model (same eval windows)
        ovls = [o for o in model.overlays.values() if getattr(o, "prior_note", False)]
        trained = [o.read_mode for o in ovls]
        ra = {}
        for mode in modes:
            for o in ovls:
                o.read_mode = mode
            ev3 = evaluate(model, eval_batches, device, detail=True)
            ra[str(mode)] = dict(loss=ev3["loss"], loss_by_type=ev3.get("loss_by_type"))
            log(dict(read_ablation=mode, **ra[str(mode)]))
        for o, m in zip(ovls, trained):
            o.read_mode = m
        extra["read_ablation"] = ra
    for name, t2 in extra_tasks.items():
        _, fn2 = make_task(t2)
        rng2 = np.random.default_rng(cfg["eval_seed"] + 1)
        n2, eb = cfg.get("extra_eval_size", 256), cfg.get("extra_eval_batch", 16)
        bs = []
        for i in range(0, n2, eb):
            b = fn2(min(eb, n2 - i), rng2)
            b["sched"] = build_schedule(b["tokens"], seed=cfg["eval_seed"] + 7 + i, **scfg)
            bs.append(b)
        ev2 = evaluate(model, bs, device)
        extra[f"acc_{name}"] = ev2["acc"]
        if has_overlay:
            extra[f"acc_{name}_ablate_q"] = evaluate(model, bs, device, ablate="query", detail=False)["acc"]
            extra[f"acc_{name}_miss_q"] = evaluate(model, bs, device, ablate="miss", detail=False)["acc"]
        log(dict(extra_eval=name, task=t2, **ev2))

    recs = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    evals = [r for r in recs if "acc" in r and "step" in r]
    if escape is None:
        escape = next((r["step"] for r in evals if r["acc"] >= 0.8), None)
    last = evals[-1] if evals else {}
    final = dict(final=True, run_id=run_id, arm=cfg["arm"], seed=cfg["seed"], lr=cfg["lr"],
                 acc=last.get("acc"), acc_clean=last.get("acc_clean"), acc_corrupt=last.get("acc_corrupt"),
                 acc_ablate=last.get("acc_ablate"), acc_ablate_q=last.get("acc_ablate_q"), acc_miss_q=last.get("acc_miss_q"),
                 escape_step=escape, wall=last.get("wall"),
                 gpu_seconds=last.get("gpu_seconds"), sec_per_step=last.get("sec_per_step"),
                 read_correct=rc, n_params=n_params(model), **extra)
    log(final)
    return final
