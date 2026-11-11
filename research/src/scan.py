"""Segmented parallel scan for r_j = a_j * r_{j-1} + b_j (plan.md §4.5).

Combine (a1, b1) then (a2, b2) -> (a2*a1, a2*b1 + b2). Hillis-Steele, inclusive,
segments given as contiguous runs (seg_start[i] = first index of i's segment).
Slots start at zero, so the scanned b is the slot state after each event.
"""
from __future__ import annotations

import torch


def segmented_scan(a: torch.Tensor, b: torch.Tensor, seg_start: torch.Tensor, max_seg_len: int | None = None):
    """a: [E, 1] or [E, d]; b: [E, d]; seg_start: [E] long. Returns states [E, d]."""
    E = b.shape[0]
    if E == 0:
        return b
    idx = torch.arange(E, device=b.device)
    limit = E if max_seg_len is None else max_seg_len
    off = 1
    while off < limit:
        same = ((idx - off) >= seg_start).unsqueeze(-1)
        a_prev = torch.cat([torch.ones_like(a[:off]), a[:-off]])
        b_prev = torch.cat([torch.zeros_like(b[:off]), b[:-off]])
        b = torch.where(same, a * b_prev + b, b)
        a = torch.where(same, a * a_prev, a)
        off *= 2
    return b
