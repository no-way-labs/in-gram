"""Token-by-token decoding with In-gram.

`InGramDecoder` holds the state of one session: for each cue length, a dictionary from the address of a cue
to its slot vector. Each step hashes one cue per cue length, reads one slot and one table row, and writes
one slot, whatever the length of the context. This is a plain reference implementation, written to make the
write and read rule explicit, and it reproduces the parallel forward pass of `InGram`.
"""
from __future__ import annotations

from collections import deque

import torch

from .hashing import MAX_CUE_LENGTH, hash_cue, table_row


class InGramDecoder:
    def __init__(self, layer):
        self.layer = layer
        self.slots = [dict() for _ in layer.cue_lengths]   # per cue length: address -> slot vector
        self.history = deque(maxlen=MAX_CUE_LENGTH)        # last tokens since the most recent boundary
        self.prev = [None] * len(layer.cue_lengths)        # address of the cue that ended at the previous position

    @property
    def n_slots(self) -> int:
        return sum(len(s) for s in self.slots)

    def state_bytes(self, bytes_per_value: int = 4) -> int:
        """Size of the slot vectors (the hash index is not counted)."""
        return self.n_slots * self.layer.d_mem * bytes_per_value

    @torch.no_grad()
    def step(self, x: torch.Tensor, token: int) -> torch.Tensor:
        """x [d_model]: hidden state of the new token where the layer is inserted. Returns x + y."""
        layer = self.layer
        token = int(token)
        if token in layer.boundary_ids:
            self.history.clear()
        else:
            self.history.append(token)
        xc = x if x.dtype == torch.float64 else x.float()
        tokens = list(self.history)
        current, reads = [], []
        for o, n in enumerate(layer.cue_lengths):
            # the cue of length n that ends at this position, if the context holds one
            addr = hash_cue(tokens[-n:], n) if len(tokens) >= n else None
            current.append(addr)
            if addr is None:
                reads += [xc.new_zeros(layer.d_mem), xc.new_ones(1)]
                continue
            slot = self.slots[o].get(addr) if layer.writes else None
            row = int(table_row(addr, layer.table_rows))
            r = layer.table[o, row].to(xc.dtype)
            reads += [r if slot is None else r + slot, xc.new_ones(1) if slot is None else xc.new_zeros(1)]
        y = layer.inject(xc, torch.cat(reads))
        # the write of this position goes to the cue that ended one position earlier, after the read
        if layer.writes and token != layer.pad_id:
            for o in range(len(layer.cue_lengths)):
                addr = self.prev[o]
                if addr is None:
                    continue
                v = layer.W_v[o] @ xc
                beta = torch.sigmoid(layer.w_beta[o] @ xc + layer.b_beta[o])
                s = self.slots[o].get(addr)
                self.slots[o][addr] = beta * v if s is None else s + beta * (v - s)
        self.prev = current
        return x + y.to(x.dtype)
