"""The In-gram layer (Section 2 of the paper).

For every position and every cue length the layer
  * writes a note into the slot of the cue that ends one position earlier (delta rule), and
  * reads the slot and the learned table row of the cue that ends at the position,
then adds a gated projection of the six reads to the hidden state.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .scan import segmented_scan
from .schedule import CUE_LENGTHS, build_schedule


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned gain."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + torch.finfo(x.dtype).eps) * self.weight


class InGram(nn.Module):
    """In-Context N-gram Memory.

    d_model: width of the hidden state. cue_lengths: the N-gram lengths, one memory each. d_mem: width of a
    slot and of a table row. table_rows: rows of the learned table per cue length. beta_bias: initial bias of
    the write strength (sigmoid(2) = 0.88). writes=False disables the in-context writes, which leaves a
    static hashed N-gram table (the control of Section 4.2).

    Use `schedule(tokens)` to compute the write and read schedule of a batch, then `forward(x, schedule)`
    with the hidden states x [B, L, d_model] at the place where the layer is inserted.
    """

    def __init__(self, d_model: int, cue_lengths=CUE_LENGTHS, d_mem: int = 64, table_rows: int = 1 << 16,
                 beta_bias: float = 2.0, writes: bool = True, boundary_ids=(), pad_id: int | None = None):
        super().__init__()
        self.d_model, self.cue_lengths, self.d_mem, self.table_rows = d_model, tuple(cue_lengths), d_mem, table_rows
        self.writes, self.boundary_ids, self.pad_id = writes, tuple(boundary_ids), pad_id
        O = len(self.cue_lengths)
        self.W_v = nn.Parameter(torch.randn(O, d_mem, d_model) * 0.02)      # note: v = W_v x
        self.w_beta = nn.Parameter(torch.randn(O, d_model) * 0.02)          # write strength: sigmoid(w x + b)
        self.b_beta = nn.Parameter(torch.full((O,), float(beta_bias)))
        self.table = nn.Parameter(torch.randn(O, table_rows, d_mem) * 0.02)  # learned rows R
        width = O * (d_mem + 1)                                              # six reads, each with a miss flag
        self.W_K = nn.Linear(width, d_model, bias=False)
        self.W_O = nn.Linear(width, d_model, bias=False)
        nn.init.normal_(self.W_K.weight, std=0.02)
        nn.init.normal_(self.W_O.weight, std=0.02)
        self.norm_x, self.norm_k = RMSNorm(d_model), RMSNorm(d_model)

    def schedule(self, tokens, device=None) -> dict:
        """Write and read schedule of a batch of token ids [B, L], as tensors on `device`."""
        tokens = tokens.detach().cpu().numpy() if isinstance(tokens, torch.Tensor) else np.asarray(tokens)
        s = build_schedule(tokens, self.cue_lengths, self.table_rows, self.boundary_ids, self.pad_id)
        device = device if device is not None else self.table.device
        return {k: (torch.as_tensor(v, device=device) if isinstance(v, np.ndarray) else v) for k, v in s.items()}

    def read(self, x: torch.Tensor, sched: dict) -> torch.Tensor:
        """The reads m [B, L, O * (d_mem + 1)]: for each cue length, R + S followed by the miss flag."""
        B, L, D = x.shape
        O, dm = len(self.cue_lengths), self.d_mem
        xf = x.reshape(B * L, D)
        ptr = sched["read_ptr"].reshape(-1)                                  # [B * L * O]
        E = sched["ev_flat"].shape[0]
        if not self.writes or E == 0:
            hit = torch.zeros_like(ptr, dtype=torch.bool).unsqueeze(-1)
            slot = xf.new_zeros(ptr.shape[0], dm)
        else:
            xe = xf[sched["ev_flat"]]                                        # hidden state at each write
            cue = sched["ev_cue"]
            v = torch.einsum("emd,ed->em", self.W_v[cue], xe)
            beta = torch.sigmoid((self.w_beta[cue] * xe).sum(-1, keepdim=True) + self.b_beta[cue].unsqueeze(-1))
            # S <- S + beta (v - S) = (1 - beta) S + beta v, scanned within each slot from S = 0
            states = segmented_scan((1.0 - beta).expand(E, dm), beta * v, sched["ev_seg_start"], int(sched["max_seg_len"]))
            hit = (ptr >= 0).unsqueeze(-1)
            slot = torch.where(hit, states[ptr.clamp(min=0)], 0.0)
        row = sched["read_row"].reshape(-1)
        cue_of_read = torch.arange(ptr.shape[0], device=x.device) % O
        table = self.table.reshape(O * self.table_rows, dm)
        R = torch.where((row >= 0).unsqueeze(-1), table[cue_of_read * self.table_rows + row.clamp(min=0)].to(x.dtype), 0.0)
        m = torch.cat([R + slot, (~hit).to(x.dtype)], dim=-1)
        return m.reshape(B, L, O * (dm + 1))

    def inject(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """y = alpha * W_O m with alpha = sigmoid(<norm(x), norm(W_K m)> / sqrt(d))."""
        k = self.norm_k(self.W_K(m))
        alpha = torch.sigmoid((self.norm_x(x) * k).sum(-1, keepdim=True) / math.sqrt(x.shape[-1]))
        return alpha * self.W_O(m)

    def forward(self, x: torch.Tensor, sched: dict) -> torch.Tensor:
        """x [B, L, d_model] -> x + y. The memory is computed in 32-bit floats under autocast."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            xc = x if x.dtype == torch.float64 else x.float()
            y = self.inject(xc, self.read(xc, sched))
        return x + y.to(x.dtype)
