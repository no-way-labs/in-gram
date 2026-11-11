"""Schedule of writes and reads, computed from token ids before the forward pass.

For each sequence and each cue length:
  * the write at position p goes to the slot of the cue that ends at p - 1, with a note computed from the
    hidden state at p;
  * the read at position t takes the slot of the cue that ends at t and sees the writes of positions p < t.

Because the addresses depend on token ids alone, the events can be sorted by slot and position on the CPU,
and each read receives a pointer to the last write that precedes it.
"""
from __future__ import annotations

import numpy as np

from .hashing import cue_addresses, table_row

CUE_LENGTHS = (1, 2, 4, 8, 16, 32)


def build_schedule(tokens, cue_lengths=CUE_LENGTHS, table_rows: int = 1 << 16, boundary_ids=(), pad_id: int | None = None) -> dict:
    """tokens: integer array [B, L].

    boundary_ids: token ids that no cue may contain. pad_id: a token id whose positions write nothing.
    Returns numpy arrays (E = number of write events, O = number of cue lengths):
      ev_flat      [E]        b * L + p of each write, sorted by (sequence, cue length, slot, position)
      ev_cue       [E]        index of the cue length of each write
      ev_seg_start [E]        index of the first write to the same slot
      max_seg_len  int        largest number of writes to one slot
      read_ptr     [B, L, O]  index of the last write that the read sees, or -1
      read_row     [B, L, O]  row of the learned table, or -1 where the position has no cue
      n_slots      [B]        slots written per sequence
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    B, L = tokens.shape
    O = len(cue_lengths)
    addr = np.zeros((B, L, O), dtype=np.int64)
    valid = np.zeros((B, L, O), dtype=bool)
    for o, n in enumerate(cue_lengths):
        addr[..., o], valid[..., o] = cue_addresses(tokens, n, boundary_ids)
    w_addr = np.zeros_like(addr)
    w_valid = np.zeros_like(valid)
    w_addr[:, 1:] = addr[:, :-1]
    w_valid[:, 1:] = valid[:, :-1]
    if pad_id is not None:
        w_valid &= (tokens != pad_id)[:, :, None]

    eb, ep, eo = np.nonzero(w_valid)
    rb, rs, ro = np.nonzero(valid)
    n_reads = len(rb)
    group = np.concatenate([rb * O + ro, eb * O + eo])
    address = np.concatenate([addr[rb, rs, ro], w_addr[eb, ep, eo]])
    position = np.concatenate([rs, ep])
    is_write = np.concatenate([np.zeros(n_reads, np.int8), np.ones(len(eb), np.int8)])  # read first at equal position
    order = np.lexsort((is_write, position, address, group))
    g, a, w = group[order], address[order], is_write[order] == 1
    new_slot = np.ones(len(order), dtype=bool)
    new_slot[1:] = (g[1:] != g[:-1]) | (a[1:] != a[:-1])
    slot = np.cumsum(new_slot) - 1
    write_rank = np.cumsum(w) - 1
    last_write = np.maximum.accumulate(np.where(w, write_rank, -1)) if len(order) else np.zeros(0, np.int64)
    ev_slot = slot[w]
    seen = last_write >= 0
    seen[seen] = ev_slot[last_write[seen]] == slot[seen]
    ptr = np.where(seen, last_write, -1)

    read_ptr = np.full((B, L, O), -1, dtype=np.int64)
    r = order[~w]
    read_ptr[rb[r], rs[r], ro[r]] = ptr[~w]
    e = order[w] - n_reads
    ev_b, ev_pos, ev_cue = eb[e], ep[e], eo[e]
    E = len(e)
    if E:
        first = np.ones(E, dtype=bool)
        first[1:] = ev_slot[1:] != ev_slot[:-1]
        idx = np.arange(E)
        seg_start = np.maximum.accumulate(np.where(first, idx, 0))
        max_seg_len = int((idx - seg_start).max()) + 1
        n_slots = np.bincount(ev_b[first], minlength=B)
    else:
        seg_start, max_seg_len, n_slots = np.zeros(0, np.int64), 0, np.zeros(B, np.int64)
    return dict(
        ev_flat=ev_b * L + ev_pos,
        ev_cue=ev_cue,
        ev_seg_start=seg_start,
        max_seg_len=max_seg_len,
        read_ptr=read_ptr,
        read_row=np.where(valid, table_row(addr, table_rows), -1),
        n_slots=n_slots,
    )
