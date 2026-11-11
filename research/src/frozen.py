"""Frozen pretrained backbone + a trained bolt-on module (exact document memory, EDM, or a
parameter-matched adapter control), inserted after chosen decoder layers via forward hooks.

Used for the attention-ladder experiment: m-a-p 1.3B GatedDeltaNet models with 0 / 1 / 2 / 4 / 8
attention layers out of 24, each frozen as released, + adapter, + EDM. EDM's value is EDM minus
adapter; its attention-equivalent is where pure GDN + EDM meets the ladder curve.

EDM = the bolted-at-depth locker that beat the unified head (lm-bolt): `Overlay` reading and
writing the hidden states at the insertion point, 1-5-gram cues, H = 1, mean write, d_mem 256.

Works with any model exposing a decoder-layer list at `layers_attr` (default "model.layers", the
Hugging Face / fla convention) whose forward takes input_ids and returns logits.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .overlay import Overlay
from .schedule import build_schedule


def _get(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


class Adapter(nn.Module):
    """Control: bottleneck MLP added to the residual stream, zero-initialized output (starts as
    identity), no memory. Same call signature as Overlay."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.down = nn.Linear(d_model, hidden)
        self.up = nn.Linear(hidden, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, sched=None, ablate=False):
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = self.up(F.gelu(self.down(self.norm(x.float()))))
        return x + y.to(x.dtype), {}


def n_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def make_module(kind: str, d_model: int, cfg: dict) -> nn.Module:
    if kind == "edm":
        return Overlay(d_model=d_model, orders=tuple(cfg.get("orders", (1, 2, 3, 4, 5))), H=1,
                       d_mem=cfg.get("d_mem", 256), write_mode=cfg.get("write_mode", "mean"))
    if kind == "adapter":  # match EDM's parameter count
        target = n_trainable(make_module("edm", d_model, cfg))
        hidden = max(8, round((target - d_model) / (2 * d_model + 1)))
        return Adapter(d_model, hidden)
    raise ValueError(kind)


class FrozenWithMemory(nn.Module):
    """Freeze `base`; add one trainable module after each decoder layer in `insert_at`."""

    def __init__(self, base: nn.Module, kind: str | None, insert_at=(2,), cfg: dict | None = None,
                 layers_attr: str = "model.layers"):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.kind = kind
        self.mods = nn.ModuleDict()
        self._sched = None
        layers = _get(base, layers_attr)
        d_model = base.config.hidden_size
        if kind:
            for i in insert_at:
                self.mods[str(i)] = make_module(kind, d_model, cfg or {})
                layers[i].register_forward_hook(self._hook(str(i)))

    def _hook(self, key):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h2, _ = self.mods[key](h, self._sched)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    def forward(self, input_ids, sched=None):
        self._sched = sched
        out = self.base(input_ids=input_ids, use_cache=False)
        self._sched = None
        return out.logits if hasattr(out, "logits") else out[0]


def token_nll(logits, tokens):
    """Per-position NLL of tokens[:, 1:] given logits[:, :-1], float32. [B, L-1]."""
    lg = logits[:, :-1].float()
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tokens[:, 1:].reshape(-1), reduction="none").view(
        tokens.shape[0], -1)


def model_ids(w: np.ndarray, cfg: dict) -> np.ndarray:
    """Loader ids are shifted by cfg["id_offset"] (src/agentic.SHIFT = 2; 0 = PAD, 1 = BOS/SEP);
    a pretrained backbone needs its own native ids, and SEP maps to its BOS."""
    off = cfg.get("id_offset", 0)
    if not off:
        return w
    return np.where(w < off, cfg.get("bos_id", 1), w - off)


def _sched(tokens: np.ndarray, cfg: dict, device):
    from .train import sched_to_torch

    s = build_schedule(tokens, orders=tuple(cfg.get("orders", (1, 2, 3, 4, 5))), H=1)
    return sched_to_torch(s, device)


@torch.no_grad()
def evaluate(model: FrozenWithMemory, windows: list[np.ndarray], labels: list[np.ndarray], cfg: dict, device):
    """Mean NLL overall and by pointer regime (labels per position: 0 none, 1 short, 2 long; aligned
    with the predicted token, i.e. labels[t] describes tokens[t])."""
    model.eval()
    tot = np.zeros(3)
    cnt = np.zeros(3)
    allsum = allcnt = 0.0
    for w, lab in zip(windows, labels):
        t = torch.as_tensor(model_ids(w[None], cfg), device=device)
        s = _sched(w[None], cfg, device) if model.kind == "edm" else None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            nll = token_nll(model(t, s), t)[0].cpu().numpy()
        lab = np.asarray(lab)[1:]
        for r in range(3):
            m = lab == r
            tot[r] += nll[m].sum()
            cnt[r] += m.sum()
        allsum += nll.sum()
        allcnt += len(nll)
    model.train()
    names = ("none", "short", "long")  # plain floats: the result is logged as JSON
    return dict(loss=float(allsum / max(allcnt, 1)),
                by_regime={k: (float(tot[i] / cnt[i]) if cnt[i] else None) for i, k in enumerate(names)},
                frac_regime={k: float(cnt[i] / max(cnt.sum(), 1)) for i, k in enumerate(names)})


def train_frozen(model: FrozenWithMemory, batches, cfg: dict, device, log=None):
    """batches: iterable of int arrays [B, L]. Trains only the inserted modules."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 1e-3), weight_decay=0.0)
    steps, warm = cfg["steps"], cfg.get("warmup", 100)
    t0 = time.time()
    for step, b in enumerate(batches, 1):
        lr = cfg.get("lr", 1e-3) * min(1.0, step / warm) * 0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr
        t = torch.as_tensor(model_ids(b, cfg), device=device)
        s = _sched(b, cfg, device) if model.kind == "edm" else None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = token_nll(model(t, s), t).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if log and (step % cfg.get("log_every", 50) == 0 or step == steps):
            log(dict(step=step, train_loss=float(loss), lr=lr, sec_per_step=(time.time() - t0) / step))
        if step >= steps:
            break


def run(cfg: dict, out_dir: str | Path) -> dict:
    """One ladder arm: load cfg["base"] (HF id), insert cfg["kind"] (None / "adapter" / "edm"),
    train on agentic windows, evaluate at each cfg["eval_lengths"]. Logs to <out_dir>/<run_id>.jsonl."""
    import fla.models  # noqa: F401  (registers the fla model classes with transformers)
    from transformers import AutoModelForCausalLM

    from . import agentic

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / f"{cfg['run_id']}.jsonl").open("w")

    def log(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0))
    base = AutoModelForCausalLM.from_pretrained(cfg["base"], torch_dtype=torch.bfloat16).to(device)
    model = FrozenWithMemory(base, cfg.get("kind"), insert_at=cfg.get("insert_at", (2,)), cfg=cfg).to(device)
    for m in model.mods.values():
        m.float()
    cfg = dict(cfg, id_offset=agentic.SHIFT, bos_id=getattr(base.config, "bos_token_id", None) or 1)
    if cfg.get("grad_ckpt") and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
    log(dict(config=cfg, n_trainable=n_trainable(model)))
    ckw = dict(tokenizer=cfg["base"], max_rows=cfg.get("max_rows"))

    def train_batches():
        it = agentic.windows("train", cfg["train_len"], cfg["batch"] * cfg["steps"], cfg.get("seed", 0), **ckw)
        while True:
            b = [next(it, None) for _ in range(cfg["batch"])]
            if b[-1] is None:
                return
            yield np.stack(b)

    if cfg.get("kind"):
        train_frozen(model, train_batches(), cfg, device, log)
    res = {}
    for L in cfg.get("eval_lengths", (2048,)):  # same eval windows for every arm (fixed seed)
        ws = list(agentic.windows("eval", L, cfg.get("eval_n", 64), 4242, **ckw))
        labs = [agentic.pointer_labels(w)[1] for w in ws]
        res[str(L)] = evaluate(model, ws, labs, cfg, device)
        log(dict(eval_len=L, **res[str(L)]))
    final = dict(final=True, run_id=cfg["run_id"], kind=cfg.get("kind"), base=cfg["base"],
                 n_trainable=n_trainable(model), eval=res)
    log(final)
    logf.close()
    return final
