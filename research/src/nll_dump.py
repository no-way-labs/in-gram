"""Per-token NLL of saved checkpoints on a fixed set of eval windows (for per-tag losses).

dump(cfg): cfg = {"ckpts": [path, ...], "out": dir, "task": task dict, "n": windows, "batch": int,
"eval_seed": int}. Builds the eval windows exactly as train.py does (same seed, same order), runs
each model, and writes <out>/<checkpoint stem>.npz with nll float16 [n, L] (position 0 = nan, unscored).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .models import Model
from .schedule import build_schedule
from .tasks import IGNORE, make_task
from .train import head_chunks, sched_to_torch, with_lmax


def eval_batches(task: dict, n: int, batch: int, scfg: dict, eval_seed: int = 987654):
    task = with_lmax(task)  # held-out split for real data, as train.py does; synthetic tasks have none
    _, fn = make_task({**task, "split": "validation"} if task.get("name") in ("lm", "agentic") else task)
    rng = np.random.default_rng(eval_seed)
    out = []
    for i in range(0, n, batch):
        b = fn(min(batch, n - i), rng)
        b["sched"] = build_schedule(b.get("hash_tokens", b["tokens"]), seed=eval_seed + i, **scfg)
        out.append(b)
    return out


@torch.no_grad()
def model_nll(model, batches, device) -> np.ndarray:
    model.eval()
    rows = []
    for b in batches:
        tokens = torch.as_tensor(b["tokens"], device=device)
        targets = torch.as_tensor(b["targets"], device=device)
        sched = sched_to_torch(b["sched"], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h, _ = model(tokens, sched, return_hidden=True)
        mask = targets != IGNORE
        nll = torch.full(targets.shape, float("nan"), device=device)
        parts, off, tg = [], 0, targets[mask]
        for lg in head_chunks(model, h[mask], device):
            parts.append(F.cross_entropy(lg, tg[off:off + lg.shape[0]], reduction="none"))
            off += lg.shape[0]
        nll[mask] = torch.cat(parts) if parts else tg.float()
        rows.append(nll.float().cpu().numpy())
    return np.concatenate(rows, 0).astype(np.float16)


def attn_probs(mixer, h, rows):
    """Softmax attention probabilities [B, H, len(rows), L] of a CausalAttention for query rows,
    recomputed from its own projections (qkv_heads), causal."""
    q, k, _ = mixer.qkv_heads(h)
    q, k = q.float(), k.float()
    sc = torch.einsum("bhtd,bhsd->bhts", q[:, :, rows], k) / q.shape[-1] ** 0.5
    L = k.shape[2]
    t = torch.arange(rows.start, rows.stop, device=h.device).view(-1, 1)
    sc = sc.masked_fill(t < torch.arange(L, device=h.device).view(1, -1), float("-inf"))
    return torch.softmax(sc, -1)


@torch.no_grad()
def attn_stats(model, batches, device, far: int = 512, chunk: int = 512) -> dict:
    """Per (layer, head): mean attention mass beyond `far` tokens, and the parts of it on induction
    positions (key j right after an earlier occurrence of the query's 2-token context:
    tokens[j-1] = tokens[i], tokens[j-2] = tokens[i-1]) and on copies of the answer (tokens[j] =
    target[i]). Averaged over queries with i > far. Full (non-windowed) CausalAttention layers only."""
    from .models import CausalAttention

    model.eval()
    layers = [(i, b.mixer) for i, b in enumerate(model.blocks)
              if isinstance(b.mixer, CausalAttention) and not b.mixer.window]
    if not layers:
        return {}
    grabbed = {}
    hooks = [m.register_forward_pre_hook(lambda mod, args, i=i: grabbed.__setitem__(i, args[0].detach()))
             for i, m in layers]
    acc = {}
    try:
        for b in batches:
            tokens = torch.as_tensor(b["tokens"], device=device)
            targets = torch.as_tensor(b["targets"], device=device)
            sched = sched_to_torch(b["sched"], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(tokens, sched, return_hidden=True)
            B, L = tokens.shape
            prev1 = torch.roll(tokens, 1, 1)
            prev2 = torch.roll(tokens, 2, 1)
            j = torch.arange(L, device=device)
            for i, mixer in layers:
                h = grabbed[i].float()
                for c0 in range(far + 1, L, chunk):
                    rows = slice(c0, min(L, c0 + chunk))
                    p = attn_probs(mixer, h, rows)  # [B, H, C, L]
                    ti = torch.arange(rows.start, rows.stop, device=device)
                    farm = (ti.view(-1, 1) - j.view(1, -1)) > far  # [C, L]
                    ind = (prev1.unsqueeze(1) == tokens[:, rows].unsqueeze(2)) & \
                          (prev2.unsqueeze(1) == prev1[:, rows].unsqueeze(2)) & (j >= 2).view(1, 1, -1)
                    cp = tokens.unsqueeze(1) == targets[:, rows].unsqueeze(2)  # [B, C, L]
                    valid = (targets[:, rows] >= 0).float()  # [B, C]
                    def mass(m):  # noqa: E306
                        return ((p * m.unsqueeze(1).float()).sum(-1) * valid.unsqueeze(1)).sum((0, 2))  # [H]
                    a = acc.setdefault(i, dict(n=0.0, far=0, far_ind=0, far_copy=0, ind=0, copy=0))
                    a["far"] = a["far"] + mass(farm.unsqueeze(0).expand_as(cp))
                    a["far_ind"] = a["far_ind"] + mass(farm.unsqueeze(0) & ind)
                    a["far_copy"] = a["far_copy"] + mass(farm.unsqueeze(0) & cp)
                    a["ind"] = a["ind"] + mass(ind)
                    a["copy"] = a["copy"] + mass(cp)
                    a["n"] += float(valid.sum())
    finally:
        for hk in hooks:
            hk.remove()
    return {str(i): {k: (v / a["n"]).tolist() for k, v in a.items() if k != "n"} | {"queries": a["n"]}
            for i, a in acc.items()}


def dump(cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    done = {}
    for p in cfg["ckpts"]:
        p = Path(p)
        if not p.exists():
            done[p.stem] = dict(error="missing")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model = Model(ck["mcfg"]).to(device)
        model.load_state_dict(ck["model"])
        batches = eval_batches(cfg["task"], cfg["n"], cfg["batch"], ck["scfg"], cfg.get("eval_seed", 987654))
        nll = model_nll(model, batches, device)
        np.savez_compressed(out / f"{p.stem}.npz", nll=nll)
        done[p.stem] = dict(mean_nll=float(np.nanmean(nll.astype(np.float32))))
        if cfg.get("attn_stats"):
            done[p.stem]["attn"] = attn_stats(model, batches, device, far=cfg.get("far", 512))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return dict(ok=True, models=done)
