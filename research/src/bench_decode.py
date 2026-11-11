"""Decode benchmark: tokens/s, peak memory and state bytes after a long prefill (batch 1).

For each arm and context length L: prefill L tokens (parallel forward, then load the decode state;
if that runs out of memory, a timing-only synthetic state of the same size), then greedy-decode
`warmup` + `n_decode` tokens one at a time through Decoder.step and time the last n_decode.
Untrained weights: timing and memory only. An OOM is recorded, and longer lengths for that arm
are skipped. Entry points: bench(cfg) (Rhombic kind "bench_decode") and scripts/bench_decode.py.
"""
from __future__ import annotations

import time
import traceback

import numpy as np
import torch

from .controls import arm_config
from .models import Model, n_params

# Same values as scripts/launch_rhombic.py (LM_D256, ROPE_ATTN, ATTN_CONV, HYBRID4, O15_MIX).
LM_D256 = {"d_model": 256, "n_blocks": 4, "heads": 4, "mlp_hidden": 512}
ROPE_ATTN = {"rope": True, "pos_emb": False}
ATTN_CONV = {"attn_conv": 4}
HYBRID4 = ["gdn", "gdn", "gdn", "attn"]
O15_MIX = {"overlay.orders": [1, 2, 3, 4, 5] * 2, "overlay.H": 1,
           "overlay.write_mode": ["mean"] * 5 + ["delta"] * 5}
DEFAULT_ARMS = {
    "attn": dict(arm="attn", overrides={**ROPE_ATTN, **ATTN_CONV}),
    "gdn": dict(arm="base", overrides={}),
    "gdn_locker": dict(arm="ovl", overrides={**O15_MIX, "static_rows": 1 << 17}),
    "hybrid": dict(arm="base", overrides={"mixer": HYBRID4, **ROPE_ATTN, **ATTN_CONV}),
    "hybrid_locker": dict(arm="ovl", overrides={"mixer": HYBRID4, **ROPE_ATTN, **ATTN_CONV, **O15_MIX,
                                                "static_rows": 1 << 17}),
}
DEFAULTS = dict(lengths=[4096, 16384, 65536, 262144, 1048576], n_decode=256, warmup=16,
                tokens="wikitext", seed=0, **LM_D256)


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def _token_source(cfg, max_len):
    """(vocab, fn(L) -> int64 [1, L] starting with BOS)."""
    from .tasks import BOS

    if cfg["tokens"] == "wikitext":
        try:
            from . import text

            stream = text.load_tokens("wikitext-103-raw-v1", "train")
            vocab = text.vocab_size()
            off = int(np.random.default_rng(cfg["seed"]).integers(0, len(stream) - max_len))

            def get(L):
                return np.concatenate([[BOS], np.asarray(stream[off:off + L - 1], dtype=np.int64)])[None]

            return vocab, get, "wikitext-103 train"
        except Exception as e:  # fall back to random tokens, and say so
            why = f"{type(e).__name__}: {e}"
    else:
        why = None
    vocab = cfg.get("vocab") or 50259
    rng = np.random.default_rng(cfg["seed"])
    seq = rng.integers(2, vocab, size=max_len)
    seq[0] = BOS

    def get(L):
        return seq[None, :L].astype(np.int64)

    return vocab, get, "random" + (f" (wikitext failed: {why})" if why else "")


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _peak_gb(dev):
    return round(torch.cuda.max_memory_allocated() / 2**30, 3) if dev.type == "cuda" else None


def _reset_peak(dev):
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def bench_one(model, scfg, toks, cfg, dev) -> dict:
    L = toks.shape[1]
    ac = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    dec = model.decoder(batch=1, scfg=scfg, max_len=L + cfg["warmup"] + cfg["n_decode"])
    tt = torch.as_tensor(toks, device=dev)
    out = dict(L=L)
    _reset_peak(dev)
    t0 = time.time()
    logits = None
    if cfg.get("prefill") != "synthetic":  # "synthetic" forces the timing-only state (tests)
        try:
            with torch.autocast(**ac):
                logits = dec.prefill(tt)
            out["prefill"] = "parallel"
        except Exception as e:
            if not _is_oom(e):
                raise
            out["prefill_parallel_oom"] = True
    if logits is None:  # outside the except block, so the failed forward's tensors are freed
        dec.reset()
        _reset_peak(dev)
        t0 = time.time()
        oom = False
        try:
            with torch.autocast(**ac):
                dec.prefill_synthetic(tt[:, :-1])
                logits = dec.step(tt[:, -1])  # the last prefix token, to get logits
            out["prefill"] = "synthetic"
        except Exception as e2:
            if not _is_oom(e2):
                raise
            oom = True
        if oom:
            dec.reset()
            return dict(out, oom="prefill")
    _sync(dev)
    out["prefill_s"] = round(time.time() - t0, 3)
    out["peak_gb_prefill"] = _peak_gb(dev)
    nxt = int(logits.argmax(-1)[0])
    try:
        with torch.autocast(**ac):
            for _ in range(cfg["warmup"]):
                nxt = int(dec.step([nxt]).argmax(-1)[0])
            _sync(dev)
            _reset_peak(dev)
            t0 = time.time()
            for _ in range(cfg["n_decode"]):
                nxt = int(dec.step([nxt]).argmax(-1)[0])  # greedy: the host needs each token anyway
            _sync(dev)
        dt = time.time() - t0
    except Exception as e:
        if not _is_oom(e):
            raise
        dec.reset()
        return dict(out, oom="decode")
    out.update(decode_tok_s=round(cfg["n_decode"] / dt, 2), decode_ms_per_tok=round(1000 * dt / cfg["n_decode"], 3),
               peak_gb_decode=_peak_gb(dev), state=dec.state_bytes())
    dec.reset()
    return out


def bench(cfg: dict) -> dict:
    cfg = {**DEFAULTS, **cfg}
    arms = cfg.get("arms") or DEFAULT_ARMS
    dev = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    lengths = sorted(cfg["lengths"])
    vocab, get_tokens, src = _token_source(cfg, max(lengths))
    dims = {k: cfg[k] for k in ("d_model", "n_blocks", "heads", "mlp_hidden")}
    info = dict(device=str(dev), torch=torch.__version__, tokens=src, vocab=vocab, dims=dims,
                n_decode=cfg["n_decode"], warmup=cfg["warmup"])
    if dev.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        import fla

        info["fla"] = fla.__version__
    except Exception:
        info["fla"] = None
    results = {}
    for name, a in arms.items():
        res = results[name] = {}
        try:
            mcfg, scfg = arm_config(a["arm"], {"name": "lm", "L_max": 4096},
                                    {**dims, "vocab": vocab, **(a.get("overrides") or {})})
            torch.manual_seed(cfg["seed"])
            model = Model(mcfg).to(dev).eval()
        except Exception as e:
            for L in lengths:
                res[str(L)] = dict(error=f"{type(e).__name__}: {e}"[:2000])
            continue
        res["n_params"] = n_params(model)
        for L in lengths:
            try:
                r = bench_one(model, scfg, get_tokens(L), cfg, dev)
            except Exception:
                r = dict(error=traceback.format_exc()[-3000:])
            res[str(L)] = r
            print(name, L, {k: v for k, v in r.items() if k != "state"},
                  (r.get("state") or {}).get("total"), flush=True)
            _reset_peak(dev)
            if "oom" in r:  # longer contexts would OOM too
                break
        del model
        _reset_peak(dev)
    ok = any("decode_tok_s" in r for res in results.values() for r in res.values() if isinstance(r, dict))
    return dict(ok=ok, info=info, results=results)
