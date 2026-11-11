"""Inference-side robustness of saved checkpoints: fake quantization and generation loops.

Local users run 3-4-bit weights and int8 KV caches, so we test what each part of the model costs in
NLL when stored at low precision, and whether the copy-favoring overlay makes generation loop.

Quantization (symmetric round-to-nearest, absmax scale per group, fake-quantized in place):
  w8 / w4     every nn.Linear weight, groups of 64 along the input dim; the LM head and the token
              embedding stay at 8 bits (as GGUF Q4_K-style quants keep them higher)
  kv8 / kv4   attention K and V, per token and head (group = head dim)
  tbl8 / tbl4 the overlay's static hashed table (prior_table), per row
  st8 / st4   the overlay's in-context slot state, per stored state row (every scan output)
  local       w4 + kv8 + tbl4 + st8 (a typical local configuration)
  local4      w4 + kv4 + tbl4 + st4
The GDN recurrent state is not quantized (fla kernels; it is small and fixed-size).

Generation: the first `prompt` tokens of each eval window, `new` tokens greedy and at temperature 1,
full recompute per step. Per generation: loop (the last 128 tokens repeat with some period <= 64),
distinct 4-gram ratio, share of 8-grams copied from the prompt, and agreement with the true
continuation. The same statistics on the true continuation give the base rate.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import overlay as overlay_mod
from .models import CausalAttention, Model
from .nll_dump import eval_batches, model_nll
from .schedule import build_schedule
from .train import sched_to_torch

MODES = {
    "bf16": {},
    "w8": {"w": 8}, "w4": {"w": 4},
    "kv8": {"kv": 8}, "kv4": {"kv": 4},
    "tbl8": {"tbl": 8}, "tbl4": {"tbl": 4},
    "st8": {"st": 8}, "st4": {"st": 4},
    "local": {"w": 4, "kv": 8, "tbl": 4, "st": 8},
    "local4": {"w": 4, "kv": 4, "tbl": 4, "st": 4},
}


def fq(x: torch.Tensor, bits: int, group: int | None = None) -> torch.Tensor:
    """Symmetric fake quantization along the last dim in groups (None = whole last dim)."""
    shp = x.shape
    g = group or shp[-1]
    xf = x.float().reshape(-1, g)
    qmax = 2 ** (bits - 1) - 1
    s = xf.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    return (torch.round(xf / s).clamp(-qmax, qmax) * s).reshape(shp).to(x.dtype)


@torch.no_grad()
def apply_mode(model: Model, mode: dict):
    """Quantize weights / table in place and install K/V and slot-state hooks. Returns an undo fn
    for the hooks (weights are restored by reloading the state dict)."""
    undo = []
    if mode.get("w"):
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and m.weight.shape[-1] % 64 == 0:
                bits = 8 if m is model.head else mode["w"]
                m.weight.copy_(fq(m.weight, bits, 64))
        model.embed.weight.copy_(fq(model.embed.weight, 8))
    if mode.get("tbl"):
        for m in model.modules():
            if isinstance(m, overlay_mod.Overlay) and getattr(m, "prior_table", None) is not None:
                m.prior_table.copy_(fq(m.prior_table, mode["tbl"]))
    if mode.get("kv"):
        for m in model.modules():
            if isinstance(m, CausalAttention):
                orig = m.qkv_heads

                def wrapped(x, orig=orig, b=mode["kv"]):
                    q, k, v = orig(x)
                    return q, fq(k, b), fq(v, b)
                m.qkv_heads = wrapped
                undo.append(lambda m=m: delattr(m, "qkv_heads"))
    if mode.get("st"):
        orig_scan = overlay_mod.segmented_scan

        def scan_q(*a, b=mode["st"], **k):
            return fq(orig_scan(*a, **k), b)
        overlay_mod.segmented_scan = scan_q
        undo.append(lambda: setattr(overlay_mod, "segmented_scan", orig_scan))
    return lambda: [u() for u in undo]


def loop_flag(g: np.ndarray, tail: int = 128, pmax: int = 64) -> bool:
    t = g[-tail:]
    return any((t[p:] == t[:-p]).all() for p in range(1, pmax + 1))


def gen_stats(prompt: np.ndarray, gen: np.ndarray, ref: np.ndarray) -> dict:
    g4 = {tuple(gen[i:i + 4]) for i in range(len(gen) - 3)}
    p8 = {tuple(prompt[i:i + 8]) for i in range(len(prompt) - 7)}
    c8 = np.mean([tuple(gen[i:i + 8]) in p8 for i in range(len(gen) - 7)])
    agree = gen[:len(ref)] == ref[:len(gen)]
    first_miss = int(np.argmin(agree)) if not agree.all() else len(agree)
    return dict(loop=float(loop_flag(gen)), distinct4=len(g4) / max(1, len(gen) - 3), copy8=float(c8),
                agree=float(agree.mean()), exact_prefix=first_miss)


@torch.no_grad()
def generate(model, scfg, prompts: np.ndarray, new: int, temp: float, device, seed: int = 0):
    toks = torch.as_tensor(prompts, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    for i in range(new):
        sched = sched_to_torch(build_schedule(toks.cpu().numpy(), seed=seed + i, **scfg), device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            h, _ = model(toks, sched, return_hidden=True)
            lg = model.head(h[:, -1]).float()
        if temp == 0:
            nxt = lg.argmax(-1)
        else:
            nxt = torch.multinomial(torch.softmax(lg / temp, -1), 1, generator=gen).squeeze(-1)
        toks = torch.cat([toks, nxt[:, None]], 1)
    return toks[:, prompts.shape[1]:].cpu().numpy()


def run(cfg: dict, ckpts: dict[str, Path], progress=None) -> dict:
    """cfg: task, n, batch, modes, gen_modes, prompt, new, gen_batch. ckpts: name -> local path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for ci, (name, path) in enumerate(ckpts.items()):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = Model(ck["mcfg"]).to(device)
        batches = eval_batches(cfg["task"], cfg["n"], cfg["batch"], ck["scfg"], cfg.get("eval_seed", 987654))
        res = {"nll": {}, "gen": {}}
        for mname in cfg["modes"]:
            model.load_state_dict(ck["model"])
            undo = apply_mode(model, MODES[mname])
            t0 = time.time()
            nll = model_nll(model, batches, device).astype(np.float32)
            undo()
            res["nll"][mname] = dict(mean=float(np.nanmean(nll)), sec=time.time() - t0,
                                     per_window=np.nanmean(nll, 1).tolist())
        if cfg.get("new"):
            P, N = cfg["prompt"], cfg["new"]
            toks = np.concatenate([np.asarray(b["tokens"]) for b in batches], 0)
            prompts, refs = toks[:, :P], toks[:, P:P + N]
            for mname in cfg["gen_modes"]:
                model.load_state_dict(ck["model"])
                undo = apply_mode(model, MODES[mname])
                for temp in cfg.get("temps", (0.0, 1.0)):
                    t0 = time.time()
                    gens = np.concatenate([generate(model, ck["scfg"], prompts[i:i + cfg["gen_batch"]], N, temp,
                                                    device, seed=i)
                                           for i in range(0, len(prompts), cfg["gen_batch"])], 0)
                    st = [gen_stats(p, g, r) for p, g, r in zip(prompts, gens, refs)]
                    res["gen"][f"{mname}@T{temp:g}"] = dict(
                        {k: float(np.mean([s[k] for s in st])) for k in st[0]}, sec=time.time() - t0,
                        samples=gens[:4].tolist())
                undo()
            st = [gen_stats(p, r, r) for p, r in zip(prompts, refs)]
            res["gen"]["reference"] = {k: float(np.mean([s[k] for s in st])) for k in st[0]}
        out[name] = res
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if progress:
            progress(ci + 1, len(ckpts))
    return dict(ok=True, models=out)
