"""Backbones and full model (plan.md §5)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .overlay import FrozenTable, Overlay
from .tasks import VOCAB


MIXERS = ("attn", "gdn", "gla", "none", "locker", "sdm", "hgdn", "lema", "kda", "dsa", "swa")


def block_mixers(cfg: dict) -> list[str]:
    """Per-block mixer list. cfg["mixer"] is one name (all blocks) or a list with
    one entry per block. "locker" = a locker-only block: locker at the block's
    input (insertion point i for block i), then the MLP, no sequence mixer."""
    n = cfg.get("n_blocks", 2)
    m = cfg["mixer"]
    mixers = [m] * n if isinstance(m, str) else list(m)
    assert len(mixers) == n, f"mixer list {mixers} must have n_blocks={n} entries"
    for x in mixers:
        assert x in MIXERS, f"unknown mixer {x}"
    return mixers


def fla_status() -> str | None:
    """None if fla layers + CUDA are usable, else the reason they aren't."""
    try:
        from fla.layers import GatedDeltaNet, GatedLinearAttention  # noqa: F401
    except Exception as e:
        return f"fla layers unavailable: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    return None


def fla_available() -> bool:
    return fla_status() is None


def apply_rope(x: torch.Tensor, base: float = 10000.0, offset: int = 0) -> torch.Tensor:
    """Rotary position embedding on [B, H, L, Dh] (rotate-half form), positions offset..offset+L-1."""
    L, Dh = x.shape[-2], x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, Dh, 2, device=x.device, dtype=torch.float32) / Dh))
    ang = torch.arange(offset, offset + L, device=x.device, dtype=torch.float32)[:, None] * inv[None]  # [L, Dh/2]
    cos, sin = ang.cos().repeat(1, 2).to(x.dtype), ang.sin().repeat(1, 2).to(x.dtype)
    x1, x2 = x[..., : Dh // 2], x[..., Dh // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class CausalAttention(nn.Module):
    """Softmax attention. conv > 0 adds a causal depthwise short conv on q, k, v
    (kernel `conv`, SiLU), like GDN's built-in short conv. Off by default (spec)."""

    def __init__(self, d, heads, conv: int = 0, force_flash: bool = True, rope: bool = False,
                 rope_base: float = 10000.0, rope_dims: int | None = None, window: int = 0):
        super().__init__()
        self.heads = heads
        # rope_dims: rotate only the first rope_dims of each head (partial RoPE, as in Qwen's QSA:
        # 64 of 256); the rest of the head is position-free.
        self.rope_dims = rope_dims
        self.force_flash = force_flash
        self.rope = rope
        self.rope_base = rope_base
        # Eval-time dynamic NTK scaling (set by longeval): for L > ntk_train_len the RoPE base grows
        # by (L / ntk_train_len) ** (Dh / (Dh - 2)), so the lowest frequency spans the longer window.
        self.ntk_train_len = None
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.conv = nn.Conv1d(3 * d, 3 * d, conv, groups=3 * d, padding=conv - 1) if conv else None
        self.window = window

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv_heads(x)
        if self.window:  # sliding-window attention: each query sees its last `window` keys (itself included)
            i = torch.arange(L, device=x.device)
            band = (i.view(-1, 1) >= i.view(1, -1)) & (i.view(-1, 1) - i.view(1, -1) < self.window)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=band)
            return self.out(o.transpose(1, 2).reshape(B, L, D))
        return self._attend(q, k, v, B, L, D)

    def qkv_heads(self, x):
        """q, k, v [B, H, L, Dh] after the optional short conv and RoPE (shared with analysis code)."""
        B, L, D = x.shape
        h = self.qkv(x)
        if self.conv is not None:
            h = F.silu(self.conv(h.transpose(1, 2))[..., :L].transpose(1, 2))  # causal: trim right pad
        q, k, v = (t.contiguous() for t in h.view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4))
        if self.rope:
            base = self.rope_base
            if self.ntk_train_len and L > self.ntk_train_len:
                Dh = q.shape[-1]
                base = base * (L / self.ntk_train_len) ** (Dh / (Dh - 2))
            r = self.rope_dims or q.shape[-1]
            if r < q.shape[-1]:
                q = torch.cat([apply_rope(q[..., :r], base), q[..., r:]], -1)
                k = torch.cat([apply_rope(k[..., :r], base), k[..., r:]], -1)
            else:
                q, k = apply_rope(q, base), apply_rope(k, base)
        return q, k, v

    def _attend(self, q, k, v, B, L, D):
        dt = q.dtype
        if q.is_cuda and self.force_flash:
            # Force the FlashAttention kernel: raises if it can't run instead of silently falling
            # back to the O(L^2)-memory math path (which made 4k-token attention look ~2x heavier).
            from torch.nn.attention import SDPBackend, sdpa_kernel

            if q.dtype not in (torch.float16, torch.bfloat16):
                q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True).to(dt)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(B, L, D))


class FlaMixer(nn.Module):
    def __init__(self, kind, d, heads, head_dim=64):
        super().__init__()
        why = fla_status()
        if why:
            raise RuntimeError(f"mixer '{kind}' needs flash-linear-attention and CUDA ({why}); run it on a Rhombic GPU host")
        from fla.layers import GatedDeltaNet, GatedLinearAttention

        if kind == "kda":  # Kimi Delta Attention (Kimi K3.1's linear layers): per-channel gated delta rule
            from fla.layers import KimiDeltaAttention

            self.layer = KimiDeltaAttention(hidden_size=d, num_heads=heads, head_dim=head_dim, expand_v=1,
                                            use_short_conv=True, conv_size=4, mode="chunk")
        elif kind == "gdn":
            self.layer = GatedDeltaNet(
                hidden_size=d, num_heads=heads, head_dim=head_dim, expand_v=1.0,
                use_short_conv=True, conv_size=4, mode="chunk",
            )
        else:
            self.layer = GatedLinearAttention(
                hidden_size=d, num_heads=heads, expand_k=heads * head_dim / d,
                expand_v=heads * head_dim / d, mode="chunk",
            )

        self.kind = kind
        self.layer.layer_idx = 0  # each mixer keeps its own one-layer cache for segmented inference
        # Inference on very long inputs runs in segments of this many tokens, carrying the conv and
        # recurrent state through fla's Cache: fla's fused short-conv kernel fails ("Triton Error
        # [CUDA]: invalid argument", a launch-grid limit) somewhere between 1M and 2M tokens.
        self.segment = 1 << 18

    def forward(self, x):
        if self.kind in ("gdn", "kda") and not self.training and x.shape[1] > self.segment:
            from fla.models.utils import Cache

            cache, outs = Cache(), []
            for i in range(0, x.shape[1], self.segment):
                o, _, cache = self.layer(x[:, i:i + self.segment], past_key_values=cache, use_cache=True)
                outs.append(o)
            return torch.cat(outs, 1)
        out = self.layer(x)
        return out[0] if isinstance(out, tuple) else out


class DSAMixer(nn.Module):
    """Gated DeepSeek-Sparse-Attention-style mixer (DeepSeek-V3.2; the sparse layers of Kimi K3.1).

    Lightning indexer, as in fla's naive_dsa reference: I[t, s] = sum_j w[t, j] relu(q_I[t, j] . k_I[s]
    / sqrt(DI)), one index key per token shared across the HI indexer heads; each query keeps its
    `topk` highest-scoring causal keys (all of them when t + 1 <= topk). Main attention: multi-head,
    QK-RMSNorm, NoPE, over the selected keys only, then a sigmoid output gate (Kimi's gated DSA).
    Indexer training as in V3.2's sparse stage: its input is detached and its only loss is
    KL(p || softmax(I)) over the selected keys, p = the main attention summed over heads and
    renormalized (detached); stored in `self.aux_loss` for the trainer to add. Plain softmax MHA
    over the selected set (no MLA), so this is "DSA-style", not DeepSeek's exact layer."""

    def __init__(self, d, heads, topk=256, idx_heads=4, idx_dim=32, chunk=256):
        super().__init__()
        self.H, self.dh, self.topk, self.HI, self.DI, self.chunk = heads, d // heads, topk, idx_heads, idx_dim, chunk
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm, self.k_norm = nn.RMSNorm(self.dh), nn.RMSNorm(self.dh)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d, bias=False)
        self.iq = nn.Linear(d, idx_heads * idx_dim, bias=False)
        self.ik = nn.Linear(d, idx_dim, bias=False)
        self.iw = nn.Linear(d, idx_heads, bias=False)
        self.aux_loss = None
        self.last_indices = None  # [B, L, topk] of the last forward (analysis)

    def index_logits(self, xd, rows):
        """Index scores for query rows `rows` (a slice) against all keys: [B, len(rows), L], causal."""
        B, L, _ = xd.shape
        q = self.iq(xd[:, rows]).view(B, -1, self.HI, self.DI).float()
        k = self.ik(xd).float()
        w = self.iw(xd[:, rows]).float()
        sc = F.relu(torch.einsum("bthd,bsd->bhts", q, k) / math.sqrt(self.DI))
        logits = torch.einsum("bhts,bth->bts", sc, w)
        t = torch.arange(rows.start, rows.stop, device=xd.device).view(1, -1, 1)
        return logits.masked_fill(t < torch.arange(L, device=xd.device).view(1, 1, L), float("-inf"))

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        xd = x.detach()
        K = min(self.topk, L)
        mask = torch.zeros(B, L, L, dtype=torch.bool, device=x.device)
        idx_all, kl_sum = [], x.new_zeros((), dtype=torch.float32)
        for c0 in range(0, L, self.chunk):
            rows = slice(c0, min(L, c0 + self.chunk))
            logits = self.index_logits(xd, rows)  # [B, C, L]
            idx = logits.detach().topk(K, dim=-1).indices
            sel = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, idx, True)
            sel &= torch.isfinite(logits)  # rows with fewer than K causal keys
            mask[:, rows] = sel
            idx_all.append(idx)
            if self.training:
                with torch.no_grad():
                    sc = torch.einsum("bhtd,bhsd->bhts", q[:, :, rows].float(), k.float()) / math.sqrt(self.dh)
                    p = torch.softmax(sc.masked_fill(~sel[:, None], float("-inf")), -1).sum(1)
                    p = p / p.sum(-1, keepdim=True)
                logq = torch.log_softmax(logits.masked_fill(~sel, float("-inf")), -1)
                kl = torch.where(sel, p * (torch.log(p.clamp_min(1e-12)) - logq), 0.0)
                kl_sum = kl_sum + kl.sum()
        self.aux_loss = kl_sum / (B * L) if self.training else None
        self.last_indices = torch.cat(idx_all, 1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        o = o.transpose(1, 2).reshape(B, L, D) * torch.sigmoid(self.gate(x))
        return self.out(o)


class LemaMixer(nn.Module):
    """Latest Exact Match Attention (Brösamle 2026) as a sequence mixer, the authors' op and kernel
    (src/lema_ext). q, k are binarized codes of width `head`; each query attends to the latest
    earlier key with the same code (strictly causal), else 0. Training uses the stick-breaking
    surrogate with the hardening state (c, alpha, cap, beta) set every step by the trainer
    (`set_lema_hardening` in src/train.py); `hard` = the exact op (evaluation). Init as in their
    model: std 1/sqrt(fan_in); the trainer's residual scaling is not reproduced."""

    def __init__(self, d, head=16, out_scale=1.0):
        super().__init__()
        assert d % head == 0 and head <= 64, (d, head)
        self.H, self.dh = d // head, head
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(d))
        self.o_proj.weight.data.mul_(out_scale)  # their residual-branch scaling 1/sqrt(2 depth)
        self.alpha, self.cap, self.c, self.beta = 1.0 / math.sqrt(head), 2.0, 0.0, 4.0
        self.hard = False

    def forward(self, x):
        from .lema_ext.attention import binarize, lema, lema_hard

        B, T, D = x.shape
        sp = lambda z: z.view(B, T, self.H, self.dh).transpose(1, 2)
        zq, zk, v = sp(self.q_proj(x)), sp(self.k_proj(x)), sp(self.v_proj(x))
        if self.hard:
            y = lema_hard(zq, zk, v)
        else:
            qb, kb = binarize(zq, self.beta), binarize(zk, self.beta)
            y = lema(qb, kb, v, alpha_fwd=self.alpha, alpha_bwd=min(self.alpha, self.cap), c=self.c)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, D))


class SDMMixer(nn.Module):
    """Sparse Delta Memory (Cabannes et al. 2026; pip `sparse-delta-memory`, CC-BY-NC-4.0) as a
    sequence mixer: learned product-key sparse addressing + gated delta writes. Baseline for
    'learned vs hashed addressing'. Needs its CUDA/Triton kernels (sdm_job image)."""

    def __init__(self, d, heads, slots_per_head=1024, k=8):
        super().__init__()
        from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

        args = SparseDeltaMemoryArgs(dim=d, num_heads=heads, slots_per_head=slots_per_head,
                                     num_writes=k, num_reads=k, chunk_size=64)
        self.layer = SparseDeltaMemory(args, layer_id=0)
        self.layer.init_weights()

    def forward(self, x):
        out = self.layer(x)
        return out[0] if isinstance(out, tuple) else out


class Block(nn.Module):
    def __init__(self, d, mixer, heads, mlp_hidden, attn_conv=0, sdm_slots=1024, sdm_k=8, rope=False,
                 rope_base=10000.0, hash_cfg=None, rope_dims=None, lema_head=16, lema_out_scale=1.0, dsa_topk=256, swa_window=512):
        super().__init__()
        self.mixer = None
        self.wants_sched = mixer == "hgdn"
        if mixer == "hgdn":  # Gated DeltaNet with hashed cue columns (src/hgdn.py)
            from .hgdn import HashGDNMixer

            self.mixer = HashGDNMixer(d, heads, hash_cfg=hash_cfg)
        elif mixer == "attn":
            self.mixer = CausalAttention(d, heads, conv=attn_conv, rope=rope, rope_base=rope_base, rope_dims=rope_dims)
        elif mixer == "swa":  # sliding-window attention (RoPE and conv settings as for "attn")
            self.mixer = CausalAttention(d, heads, conv=attn_conv, rope=rope, rope_base=rope_base, rope_dims=rope_dims,
                                         window=swa_window)
        elif mixer == "dsa":
            self.mixer = DSAMixer(d, heads, topk=dsa_topk)
        elif mixer == "lema":
            self.mixer = LemaMixer(d, head=lema_head, out_scale=lema_out_scale)
        elif mixer == "sdm":
            self.mixer = SDMMixer(d, heads, slots_per_head=sdm_slots, k=sdm_k)
        elif mixer in ("gdn", "gla", "kda"):
            self.mixer = FlaMixer(mixer, d, heads)
        elif mixer != "none":
            raise ValueError(mixer)
        self.norm1 = nn.RMSNorm(d) if self.mixer is not None else None
        self.norm2 = nn.RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, d))

    def forward(self, x, sched=None):
        if self.mixer is not None:
            h = self.norm1(x)
            x = x + (self.mixer(h, sched) if self.wants_sched else self.mixer(h))
        return x + self.mlp(self.norm2(x))


class Model(nn.Module):
    """cfg keys: mixer, d_model, n_blocks, heads, mlp_hidden, max_len,
    overlay (None | dict kind=memory|frozen + Overlay kwargs),
    overlay_at (list of insertion points: 0 = on the embeddings, before block 1
    [spec default]; i = after block i). Each point gets its own locker module;
    all share one schedule, since addresses depend only on token ids."""

    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg.get("d_model", 128)
        self.cfg = cfg
        vocab = cfg.get("vocab", VOCAB)
        self.embed = nn.Embedding(vocab, d)
        mixers = block_mixers(cfg)
        # pos_emb=False -> no position embedding (NoPE); spec default is learned absolute for attn.
        use_pos = "attn" in mixers and cfg.get("pos_emb", True)
        self.pos = nn.Embedding(cfg.get("max_len", 1024), d) if use_pos else None
        ov = cfg.get("overlay")
        locker_blocks = [i for i, m in enumerate(mixers) if m == "locker"]
        if locker_blocks and not ov:
            raise ValueError("a 'locker' mixer needs an overlay config")
        # Optional static hashed table (corpus prior, Engram-style) stacked under locker; shares the
        # schedule's cues (orders, H) through sched["read_row"].
        self.static = FrozenTable(d_model=d, **cfg["static"]) if cfg.get("static") else None
        self.overlay_at = Model.locker_points(cfg)
        self.overlays = nn.ModuleDict()
        if ov:
            ov = dict(ov)
            kind = ov.pop("kind", "memory")
            n_blocks = cfg.get("n_blocks", 2)
            for at in self.overlay_at:
                assert 0 <= at <= n_blocks, f"overlay_at {at} outside 0..{n_blocks}"
                self.overlays[str(at)] = FrozenTable(d_model=d, **ov) if kind == "frozen" else Overlay(d_model=d, **ov)
            # overlay_shared_table: every placement reads the same learned corpus table (one prior_table
            # Parameter), so adding placements adds only their projections and per-sequence notes.
            if cfg.get("overlay_shared_table") and len(self.overlays) > 1:
                mods = list(self.overlays.values())
                assert all(hasattr(m, "prior_table") for m in mods), "shared table needs prior_rows"
                for m in mods[1:]:
                    m.prior_table = mods[0].prior_table
        self.blocks = nn.ModuleList(
            Block(d, "none" if m == "locker" else m, cfg.get("heads", 2), cfg.get("mlp_hidden", 2 * d),
                  attn_conv=cfg.get("attn_conv", 0), sdm_slots=cfg.get("sdm_slots", 1024),
                  sdm_k=cfg.get("sdm_k", 8), rope=cfg.get("rope", False),
                  rope_base=cfg.get("rope_base", 10000.0), hash_cfg=Model.block_hash_cfg(cfg, i),
                  rope_dims=cfg.get("rope_dims"), lema_head=cfg.get("lema_head", 16),
                  lema_out_scale=(2 * cfg.get("n_blocks", 2)) ** -0.5 if cfg.get("lema_resid_init") else 1.0,
                  dsa_topk=cfg.get("dsa_topk", 256), swa_window=cfg.get("swa_window", 512))
            for i, m in enumerate(mixers)
        )
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        # Embedding init: spec is silent; use std 0.02 (GPT-2 convention). DEVIATIONS.md D2.
        nn.init.normal_(self.embed.weight, std=cfg.get("embed_std", 0.02))
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, std=cfg.get("embed_std", 0.02))

    def forward(self, tokens, sched=None, ablate=False, return_hidden=False):
        """Returns (logits, stats), or (final normed hidden, stats) with return_hidden=True so the
        caller can apply the output head in chunks (large vocab: see train.chunked_ce)."""
        x = self.embed(tokens)
        if self.pos is not None:
            x = x + self.pos(torch.arange(tokens.shape[1], device=tokens.device))
        stats = {}

        def apply(at, x):
            key = str(at)
            if key not in self.overlays:
                return x
            x, st = self.overlays[key](x, sched, ablate=ablate)
            if not stats:
                stats.update(st)  # first locker's stats feed the standard diagnostics
            stats[f"at{at}"] = st
            return x

        st_static = None
        static_at = self.cfg.get("static_at", 0)

        def apply_static(at, x):
            nonlocal st_static
            if self.static is not None and at == static_at:  # Engram-style corpus prior
                x, st_static = self.static(x, sched, ablate=False)
            return x

        x = apply(0, apply_static(0, x))
        for i, blk in enumerate(self.blocks):
            x = apply(i + 1, apply_static(i + 1, blk(x, sched)))
        if st_static is not None:  # locker's stats stay the top-level diagnostics when present
            if not stats:
                stats.update(st_static)
            stats["static"] = st_static
        if return_hidden:
            return self.norm(x), stats
        return self.head(self.norm(x)), stats

    @staticmethod
    def block_hash_cfg(cfg: dict, i: int) -> dict | None:
        """hgdn hash config for block i: hash.layers (list of block indices) limits the hashed
        columns to those blocks; other hgdn blocks are plain GDN (orders = ())."""
        hc = cfg.get("hash")
        if not hc:
            return hc
        layers = hc.get("layers")
        if layers is None or i in layers:
            return hc
        return dict(hc, orders=())

    @staticmethod
    def locker_points(cfg: dict) -> list[int]:
        if not cfg.get("overlay"):
            return []
        mixers = block_mixers(cfg)
        locker_blocks = [i for i, m in enumerate(mixers) if m == "locker"]
        return sorted(set(cfg.get("overlay_at") or ([] if locker_blocks else [0])) | set(locker_blocks))

    def decoder(self, batch: int = 1, scfg: dict | None = None, **kw):
        """Streaming decoder: .prefill(tokens) then .step(token) -> logits (src/stream.py)."""
        from .stream import Decoder

        return Decoder(self, batch=batch, scfg=scfg, **kw)

    @property
    def overlay(self):
        """First locker module (None if the arm has none)."""
        return next(iter(self.overlays.values()), None)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
