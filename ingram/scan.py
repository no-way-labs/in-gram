"""Segmented parallel prefix scan for the affine recurrence r_j = a_j * r_{j-1} + b_j.

Each slot is one segment of consecutive events, and its state starts at zero. Two steps (a1, b1) then
(a2, b2) compose to (a2 * a1, a2 * b1 + b2), so a Hillis-Steele scan yields the state after every event in a
number of rounds logarithmic in the longest segment.
"""
from __future__ import annotations

import torch


def segmented_scan(a: torch.Tensor, b: torch.Tensor, seg_start: torch.Tensor, max_seg_len: int | None = None) -> torch.Tensor:
    """a: [E, 1] or [E, d]; b: [E, d]; seg_start: [E], index of the first event of each event's segment.

    Returns the states [E, d]: entry j is the slot state after event j.
    """
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
