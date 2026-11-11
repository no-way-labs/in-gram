"""Hashed in-context memory overlay (plan.md §4) and the frozen-table control (§6).

forward(x, sched) -> (x + y, stats). Runs in fp32 regardless of outer autocast.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hashing import ngram_addresses
from .scan import segmented_scan
from .schedule import set_and_tag
from .tasks import PAD


def _inject(x, y, ablate):
    """ablate: False (normal), True (y = 0 everywhere, the spec's ablation), or a
    bool mask [B, L] (y = 0 only where True, e.g. at scored query positions)."""
    if ablate is True:
        return x
    if ablate is not False and ablate is not None:
        y = y * (~ablate).unsqueeze(-1).to(y.dtype)
    return x + y.to(x.dtype)


def _force_miss(m, mask, d_mem, width=None):
    """Replace every read at masked positions by an empty read: zeros, flag 1 (index d_mem), and
    zeros in any extra channels (e.g. age). width = per-read width (default d_mem + 1)."""
    width = width or d_mem + 1
    B, L, F_ = m.shape
    r = m.reshape(B, L, -1, width)
    empty = torch.zeros(width, device=m.device, dtype=m.dtype)
    empty[d_mem] = 1.0
    r = torch.where(mask[:, :, None, None], empty, r)
    return r.reshape(B, L, F_)


def _take(t, idx):
    """t[idx] along dim 0 via index_select. Same values as advanced indexing, but the backward is
    index_add instead of the sorting index_put(accumulate=True), which dominated locker5's step."""
    return t.index_select(0, idx.reshape(-1)).view(*idx.shape, *t.shape[1:])


def _proj_take(lins, xf, order, pos):
    """lins[order[i]](xf[pos[i]]) for every i: one matmul over all orders, then one gather.
    All lins must share an output width."""
    O = len(lins)
    W = torch.cat([l.weight for l in lins], 0)
    b = torch.cat([l.bias for l in lins]) if lins[0].bias is not None else None
    y = F.linear(xf, W, b)  # [N, O * d_out]
    return _take(y.reshape(xf.shape[0] * O, -1), pos * O + order)


def _compute(x):
    """The locker computes in fp32 regardless of outer autocast; float64 inputs stay float64 (used by
    the streaming-equivalence tests)."""
    return x if x.dtype == torch.float64 else x.float()


class _GateInject(nn.Module):
    """alpha = sigmoid(<RMSNorm(x), RMSNorm(W_K m)> / sqrt(d)); y = alpha * W_O m.
    gate=False: alpha = 1 (no context gate; the n-gram-head baseline)."""

    def __init__(self, d_model: int, d_read: int, gate: bool = True):
        super().__init__()
        self.use_gate = gate
        self.W_K = nn.Linear(d_read, d_model, bias=False)
        self.W_O = nn.Linear(d_read, d_model, bias=False)
        self.norm_x = nn.RMSNorm(d_model)
        self.norm_k = nn.RMSNorm(d_model)
        for lin in (self.W_K, self.W_O):
            nn.init.normal_(lin.weight, std=0.02)

    def forward(self, x, m):
        if not self.use_gate:
            return self.W_O(m), torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)
        d = x.shape[-1]
        k = self.norm_k(self.W_K(m))
        alpha = torch.sigmoid((self.norm_x(x) * k).sum(-1, keepdim=True) / math.sqrt(d))
        return alpha * self.W_O(m), alpha.squeeze(-1)


class Overlay(nn.Module):
    def __init__(self, d_model=128, orders=(2,), H=2, d_mem=32, beta_bias=2.0, fixed_beta: float | None = None,
                 read_dropout: float = 0.0, hit_only: bool = False, write_mode="delta",
                 age_feature: bool = False, notes: int = 0, gate: bool = True, empty_flag: bool = True,
                 prior_rows: int | None = None, kappa_init: float = 1.0, prior_miss_only: bool = False,
                 kappa_mode: str = "row", count_feature: bool = False, backoff: bool = False,
                 prior_init: bool = False, writes: bool = True, prior_learned: bool = True,
                 prior_add: bool = False, prior_note: bool = False, read_mode: str | None = None,
                 note_merge: bool = False, ctx_key: bool = False, prior_split: bool = False,
                 value_ctx: bool = False, evid_gate: str | None = None):
        """read_dropout: in training, each position's reads become empty reads with this probability
        (keeps the backbone learning its own recall). hit_only: inject y only where some read hit.
        write_mode (one value, or one per order), how a new note merges into the slot:
          "delta"   spec: slot <- (1-b) slot + b v, scalar b = sigmoid(w.x)   (latest note dominates)
          "channel" per-channel b (vector): some channels keep the gist, others take the new note
          "mean"    b = 1/n for the n-th note: the slot is the exact running mean of all notes
        All are affine in the slot, so the parallel scan stays exact.
        age_feature: append log(1 + age) / 8 to each read, age = read pos - last write pos
          (precomputed from the schedule; lets the model judge stale notes).
        notes=K > 0: don't merge notes. Keep each locker's last K notes and read them with attention:
          query from the hidden state at the read, keys from the hidden state at each write.
          The K candidates per read are fixed by the schedule (exact, parallel). For the query to
          carry context (homographs), place the locker after a block.
        gate=False, empty_flag=False with write_mode="mean": the n-gram-head baseline (Akyurek et
          al. 2024): a uniform average of the hidden states at all earlier matches, linear
          projection, no gate, and nothing added when there is no match.
        prior_rows=R: Bayesian locker. Each cue also has a learned corpus prior (a hashed table of R
          rows per cue stream, Engram-style) and a learned confidence kappa (pseudo-count). The read
          is the posterior mean (kappa * prior + n * local_mean) / (kappa + n), n = notes written
          under this cue so far in context; a cue never seen in context reads its prior. Exact and
          parallel: local_mean is the scanned slot (write_mode "mean" makes it the plain average).
          Needs the schedule built with table_rows=R (sched["read_row"]).
          prior_miss_only: read the prior only where the cue has no note in context yet (w = 1 if
          n == 0, else 0): Engram-style backoff, local notes never mixed with the prior.
          kappa_mode: "row" = one learned kappa per hashed cue row (sparse gradients: each row
          trains only when its cue is read); "stream" = one kappa per cue stream (dense);
          "prior" = kappa = exp(w . prior_row + b), w and b per stream (dense, still per cue).
        count_feature: append log(1 + n) / 8 to each read, n = notes under this cue so far (lets the
          model tell a generic, crowded cue from a specific one, like IDF).
        backoff: longest-match backoff. Per position and hash head, only cue streams whose length
          equals the longest cue with a local note are read; shorter ones read as empty (Kneser-Ney /
          infini-gram style). Applies to local notes; a Bayesian prior still fills empty reads.
          backoff="soft": keep every read and append a flag (1 = this read is the longest local
          match), so the gate learns when to back off.
          backoff=[bool per order]: hard backoff among the flagged cue streams only (e.g. the delta
          tables of a mixed locker); unflagged streams are always read.
        prior_init=True (with prior_rows, delta/channel writes): slice D, a writable N-gram
          embedding. Each cue's slot starts at its learned row R_c (written by gradient descent over
          the corpus) and is updated in context by the delta rule S <- S + b (v - S); a cue with no
          note in this document reads R_c. One table, two writers. writes=False keeps the rows only
          (a static Engram / N-gram-embedding table); prior_learned=False fixes the rows at zero
          (plain document memory, locker). The three arms share this code path.
        prior_add=True (with prior_rows, delta/channel writes): the Delta N-gram-style alternative
          (Wu 2026): read R_c + D_c, where D_c starts at zero and is written S <- S + b (v - S)
          without reference to R_c (the write error ignores the corpus row). Same parameters as
          prior_init; only the write error differs.
        prior_note=True (with notes=K, prior_rows): locker5. Each slot keeps its last K notes; the
          corpus row R_c is one more note with a learned key, and the read attends over all of them,
          so the context picks among a cue's continuations. With equal scores the prior gets weight
          1 / (1 + n): Dirichlet smoothing with alpha = exp(prior bias). A miss reads R_c."""
        super().__init__()
        self.orders, self.H, self.d_mem = tuple(orders), H, d_mem
        self.read_dropout, self.hit_only = read_dropout, hit_only
        O = len(self.orders)
        modes = [write_mode] * O if isinstance(write_mode, str) else list(write_mode)
        assert len(modes) == O and all(m in ("delta", "channel", "mean") for m in modes), modes
        self.write_modes = modes
        self.W_v = nn.ModuleList([nn.Linear(d_model, d_mem, bias=False) for _ in range(O)])
        self.w_beta = nn.ModuleList([nn.Linear(d_model, d_mem if m == "channel" else 1) for m in modes])
        for lin in self.W_v:
            nn.init.normal_(lin.weight, std=0.02)
        for lin in self.w_beta:
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.constant_(lin.bias, beta_bias)
        self.fixed_beta = fixed_beta
        self.age_feature, self.notes = age_feature, notes
        self.empty_flag = empty_flag
        self.count_feature, self.backoff = count_feature, backoff
        if isinstance(backoff, (list, tuple)):
            assert len(backoff) == len(self.orders), backoff
            self.register_buffer("_backoff_sel", torch.tensor([bool(b) for b in backoff]), persistent=False)
            backoff = "select"
        assert backoff in (False, True, "soft", "select"), backoff
        self.backoff = backoff
        self.r_width = (d_mem + 1 + int(age_feature) + int(count_feature) + int(backoff == "soft")
                        + (d_mem if prior_split else 0))  # prior_split: R_c appended last
        from .hashing import pattern_offsets

        self.register_buffer("_cue_len", torch.tensor([len(pattern_offsets(o)) for o in self.orders]),
                             persistent=False)
        if notes:
            self.W_qn = nn.ModuleList([nn.Linear(d_model, d_mem, bias=False) for _ in range(O)])
            self.W_kn = nn.ModuleList([nn.Linear(d_model, d_mem, bias=False) for _ in range(O)])
            for lin in [*self.W_qn, *self.W_kn]:
                nn.init.normal_(lin.weight, std=0.02)
        self.gate = _GateInject(d_model, O * H * self.r_width, gate=gate)
        self.prior_rows, self.prior_miss_only = prior_rows, prior_miss_only
        self.prior_init, self.writes, self.prior_add = prior_init, writes, prior_add
        assert not (prior_init and prior_add), "prior_init and prior_add are alternatives"
        self.prior_note = prior_note
        # note_merge=True (with prior_note): the slot keeps ONE note, the delta-rule running state of
        # its values D and keys K_s (same beta), and the read is a two-way softmax between R_c and it:
        # scores q.W_pk R_c / sqrt(d) + b and q.K_s / sqrt(d) + log n. With equal key scores this gives
        # the prior weight e^b / (e^b + n), as locker5 with n notes does. Slot state 2 d_mem (vs 2 K d_mem).
        self.note_merge = note_merge
        # additive2 options (one vector of slot state, like additive):
        # prior_split=True (with prior_add): hand the gate the in-context state D_c and the corpus row R_c
        #   side by side instead of their sum, so the prior-vs-evidence weighting happens in context
        #   (the mechanism locker5's ablations found dominant). Read width + d_mem; slot state unchanged.
        # value_ctx=True: the written value also sees the cue-end state, v = W_v x_p + W_vc x_{p-1}
        #   (ctxkey's writer context, folded into the value since there is no key to hold it).
        self.prior_split, self.value_ctx = prior_split, value_ctx
        # evid_gate (with prior_add): read R_c + g * D_c, g in (0, 1)^d_mem per read, initialized near 1
        # (= additive). "ctx": g = sigmoid(W_g x + a log(1 + n) + b), x = the reader's hidden state.
        # "match": g = sigmoid((W_q x) . D_c / sqrt(d) * c + a log(1 + n) + b): the slot's own state is
        # its key, so evidence that matches the reader's context gets more weight (one-vector analogue of
        # locker5's scored read). Slot state unchanged.
        assert evid_gate in (None, "ctx", "match"), evid_gate
        self.evid_gate = evid_gate
        if evid_gate:
            assert prior_add and not prior_split, "evid_gate refines the additive read"
            O_ = len(self.orders)
            if evid_gate == "ctx":
                self.W_g = nn.ModuleList([nn.Linear(d_model, d_mem, bias=False) for _ in range(O_)])
            else:
                self.W_g = nn.ModuleList([nn.Linear(d_model, d_mem, bias=False) for _ in range(O_)])
                self.g_scale = nn.Parameter(torch.zeros(O_ * H))
            for lin in self.W_g:
                nn.init.normal_(lin.weight, std=0.02)
            self.g_count = nn.Parameter(torch.zeros(O_ * H))
            self.g_bias = nn.Parameter(torch.full((O_ * H,), 3.0))  # sigmoid(3) = 0.95: starts as additive
        if prior_split:
            assert prior_add, "prior_split refines the additive read (prior_add)"
        if value_ctx:
            self.W_vc = nn.ModuleList([nn.Linear(d_model, d_mem, bias=False) for _ in range(len(self.orders))])
            for lin in self.W_vc:
                nn.init.normal_(lin.weight, std=0.02)
        # ctx_key=True (with notes): a note's key also sees the hidden state at the end of the cue it
        # was written under (position p - 1), so the read can match the reader's context to the
        # writer's context (induction-head style), not only to the continuation's content.
        self.ctx_key = ctx_key
        if ctx_key:
            assert notes, "ctx_key needs notes"
            self.W_kc = nn.ModuleList([nn.Linear(d_model, d_mem, bias=False) for _ in range(len(self.orders))])
            for lin in self.W_kc:
                nn.init.normal_(lin.weight, std=0.02)
        if note_merge:
            assert prior_note and notes == 1, "note_merge: set notes=1, prior_note=True"
        if read_mode is not None:  # train with an ablated read (see READ_MODES)
            assert prior_note and read_mode in self.READ_MODES, read_mode
            self.read_mode = read_mode
        if prior_note:
            assert notes and prior_rows and not (prior_init or prior_add), "prior_note needs notes, prior_rows"
            self.W_pk = nn.ModuleList([nn.Linear(d_mem, d_mem, bias=False) for _ in range(O)])
            for lin in self.W_pk:
                nn.init.normal_(lin.weight, std=0.02)
            self.prior_bias = nn.Parameter(torch.zeros(O * H))
        if prior_init or prior_add or prior_note:
            assert prior_rows, "prior_init / prior_add need prior_rows"
            assert all(m in ("delta", "channel") for m in modes), "prior_init uses delta / channel writes"
            self.prior_table = nn.Parameter(torch.randn(O * H, prior_rows, d_mem) * 0.02 if prior_learned
                                            else torch.zeros(O * H, prior_rows, d_mem), requires_grad=prior_learned)
        elif prior_rows:
            # Names end in "table" so param_groups exempts them from weight decay.
            self.prior_table = nn.Parameter(torch.randn(O * H, prior_rows, d_mem) * 0.02)
            assert kappa_mode in ("row", "stream", "prior"), kappa_mode
            self.kappa_mode = kappa_mode
            if kappa_mode == "row":
                self.kappa_table = nn.Parameter(torch.full((O * H, prior_rows), math.log(kappa_init)))
            else:
                self.kappa_b = nn.Parameter(torch.full((O * H,), math.log(kappa_init)))
                if kappa_mode == "prior":
                    self.kappa_w = nn.Parameter(torch.zeros(O * H, d_mem))

    def set_backoff(self, backoff):
        """Switch hard backoff at inference (False, True, or one bool per order). "soft" changes the
        read width, so it can't be switched on a trained model."""
        assert backoff != "soft", "soft backoff changes the gate's input width"
        if isinstance(backoff, (list, tuple)):
            assert len(backoff) == len(self.orders), backoff
            self._backoff_sel = torch.tensor([bool(b) for b in backoff], device=self._cue_len.device)
            backoff = "select"
        self.backoff = backoff

    def log_kappa(self, stream, row, prior):
        """log kappa for reads in cue stream `stream` at hashed row `row` with prior vector `prior`."""
        if self.kappa_mode == "row":
            return self.kappa_table[stream, row]
        lk = self.kappa_b[stream]
        if self.kappa_mode == "prior":
            lk = lk + (prior * self.kappa_w[stream]).sum(-1)
        return lk

    def _posterior(self, vals, hit, n, row):
        """Blend the local mean with the cue's prior: w = kappa / (kappa + n) on the prior.
        n [R]: notes under the cue so far (used where hit); row [R]: hashed prior row, -1 if invalid.
        R = (..., O, H) flattened, so the cue stream of read i is i % (O * H)."""
        OH = self.prior_table.shape[0]
        row = row.reshape(-1)
        stream = torch.arange(row.shape[0], device=row.device) % OH
        valid = (row >= 0).unsqueeze(-1)
        r = row.clamp(min=0)
        prior = torch.where(valid, self.prior_table[stream, r].to(vals.dtype), 0.0)
        kappa = self.log_kappa(stream, r, prior).exp().unsqueeze(-1).to(vals.dtype)
        n = torch.where(hit, n.unsqueeze(-1), 0).to(vals.dtype)
        w = (n == 0).to(vals.dtype) if self.prior_miss_only else kappa / (kappa + n)
        return w * prior + (1.0 - w) * vals

    def _prior_rows_at(self, row):
        """R_c for reads [R] (row -1 = invalid cue -> zeros). R = (..., O, H) flattened."""
        OH = self.prior_table.shape[0]
        row = row.reshape(-1)
        stream = torch.arange(row.shape[0], device=row.device) % OH
        return torch.where((row >= 0).unsqueeze(-1), self._table_at(stream, row.clamp(min=0)), 0.0)

    def _table_at(self, stream, row):
        """prior_table[stream, row] as one index_select on the flattened table."""
        OH, rows, dm = self.prior_table.shape
        return _take(self.prior_table.reshape(OH * rows, dm), stream * rows + row)

    def _backoff(self, hit, N):
        """Longest-match backoff on hit [N*O*H, 1] (bool). Returns (hit, longest_flag or None)."""
        if not self.backoff:
            return hit, None
        O, H = len(self.orders), self.H
        hv = hit.view(N, O, H)
        lens = self._cue_len.view(1, O, 1)
        longest = torch.where(hv, lens, 0).amax(dim=1, keepdim=True)
        is_longest = (hv & (lens == longest)).reshape(-1, 1)
        if self.backoff == "soft":
            return hit, is_longest
        if self.backoff == "select":  # backoff among the selected streams only
            sel = self._backoff_sel.view(1, O, 1)
            longest_sel = torch.where(hv & sel, lens, 0).amax(dim=1, keepdim=True)
            return (hv & (~sel | (lens == longest_sel))).reshape(-1, 1), None
        return is_longest, None  # hard: shorter cues read as empty

    def _finish(self, vals, hit, n, age, row, longest_flag):
        """Read vector per (position, order, head): [R, r_width]. vals [R, d_mem] (zero on a miss),
        hit [R, 1] bool, n [R] notes under the cue so far, age [R] read pos - last write pos,
        row [R] hashed prior row (prior_rows only). Shared by the parallel and streaming paths."""
        dt = vals.dtype
        if self.prior_init:
            vals = torch.where(hit, vals, self._prior_rows_at(row).to(dt))
        elif self.prior_add and self.prior_split:
            pass  # D_c stays in vals (0 on a miss); R_c is appended last (below)
        elif self.prior_add:  # vals is the zero-started patch (0 on a miss); add the corpus row
            vals = vals + self._prior_rows_at(row).to(dt)
        elif self.prior_note:
            pass  # the prior is already one of the attended notes
        elif self.prior_rows:
            vals = self._posterior(vals, hit, n, row)
        parts = [vals, (~hit).to(dt) if self.empty_flag else torch.zeros_like(hit, dtype=dt)]
        if self.age_feature:
            parts.append(torch.where(hit, torch.log1p(age.to(dt)).unsqueeze(-1) / 8.0, 0.0))
        if longest_flag is not None:
            parts.append(longest_flag.to(dt))
        if self.count_feature:
            parts.append(torch.where(hit, torch.log1p(n.to(dt)).unsqueeze(-1) / 8.0, 0.0))
        if self.prior_add and self.prior_split:  # after the flag, so _force_miss (zero D_c, flag 1) keeps R_c
            parts.append(self._prior_rows_at(row).to(dt))
        return torch.cat(parts, -1)

    def event_values(self, x_ev: torch.Tensor, ev_order: torch.Tensor, count: torch.Tensor | None = None):
        """v [E, d_mem] and beta [E, d_mem] for each event. x_ev [E, d], ev_order [E].
        count [E]: 1-based index of the event within its slot (needed for "mean")."""
        E = x_ev.shape[0]
        ar = torch.arange(E, device=x_ev.device)
        v = torch.stack([W(x_ev) for W in self.W_v])[ev_order, ar]
        betas = []
        for mode, w in zip(self.write_modes, self.w_beta):
            if self.fixed_beta is not None:
                b = torch.full((E, 1), self.fixed_beta, device=x_ev.device, dtype=x_ev.dtype)
            elif mode == "mean":
                assert count is not None, "write_mode 'mean' needs per-event counts"
                b = (1.0 / count.to(x_ev.dtype)).unsqueeze(-1)
            else:
                b = torch.sigmoid(w(x_ev))
            betas.append(b.expand(E, self.d_mem))
        beta = torch.stack(betas)[ev_order, ar]
        return v, beta

    def _event_values_gather(self, xf, ev_flat, ev_order, count):
        """Same values as event_values(xf[ev_flat], ...), but projects per position first and
        gathers [E, d_mem] after, so no [E, d_model] copy is made (E = L*O*H events: at 1M tokens
        that copy alone is ~6 GB)."""
        E = ev_flat.shape[0]
        v = _proj_take(self.W_v, xf, ev_order, ev_flat)
        if self.value_ctx:  # a write at p has its cue ending at p - 1 >= 0 in the same row
            v = v + _proj_take(self.W_vc, xf, ev_order, ev_flat - 1)
        widths = {w.out_features for w in self.w_beta}
        if self.fixed_beta is None and "mean" not in self.write_modes and len(widths) == 1:
            beta = torch.sigmoid(_proj_take(self.w_beta, xf, ev_order, ev_flat)).expand(E, self.d_mem).contiguous()
            return v, beta
        beta = xf.new_empty(E, self.d_mem)
        for o, (mode, w) in enumerate(zip(self.write_modes, self.w_beta)):
            sel = ev_order == o
            if self.fixed_beta is not None:
                beta[sel] = self.fixed_beta
            elif mode == "mean":
                beta[sel] = (1.0 / count[sel].to(xf.dtype)).unsqueeze(-1)
            else:
                beta[sel] = torch.sigmoid(w(xf))[ev_flat[sel]].expand(-1, self.d_mem)
        return v, beta

    def read(self, x: torch.Tensor, sched: dict):
        B, L, D = x.shape
        O, H = len(self.orders), self.H
        xf = x.reshape(B * L, D)
        seg_start = sched["ev_seg_start"]
        E = seg_start.shape[0]
        count = torch.arange(E, device=x.device) - seg_start + 1
        v, beta = self._event_values_gather(xf, sched["ev_flat"], sched["ev_order"], count)
        ptr = sched["read_ptr"].reshape(-1)  # [B*L*O*H]
        if not self.writes:  # static table: every read is a miss, so it reads R_c
            ptr = torch.full_like(ptr, -1)
        hit, longest_flag = self._backoff((ptr >= 0).unsqueeze(-1), B * L)
        ptr = torch.where(hit.squeeze(-1), ptr, -1)
        p0 = ptr.clamp(min=0)
        if E == 0:
            vals = torch.zeros(ptr.shape[0], self.d_mem, device=x.device, dtype=x.dtype)
        elif self.note_merge:
            vals = self._read_merged(xf, v, beta, sched, p0, hit, B, L)
        elif self.notes:
            vals = self._read_notes(xf, v, sched, p0, hit, B, L)
        else:
            b = beta * v
            if self.prior_init:  # slot starts at R_c: first write is b v + (1 - b) R_c
                first = (torch.arange(E, device=x.device) == seg_start).unsqueeze(-1)
                rr = sched["read_row"]  # the write cue at p ends at p - 1
                ev_row = rr[sched["ev_b"], sched["ev_pos"] - 1, sched["ev_order"], sched["ev_h"]]
                stream = sched["ev_order"] * H + sched["ev_h"]
                R_ev = self._table_at(stream, ev_row.clamp(min=0)).to(b.dtype)
                b = b + torch.where(first, (1.0 - beta) * R_ev, 0.0)
            states = segmented_scan(1.0 - beta, b, seg_start, int(sched["max_seg_len"]))
            vals = torch.where(hit, _take(states, p0), 0.0)
        if E:
            n = p0 - seg_start[p0] + 1
            s_pos = torch.arange(L, device=x.device).view(1, L, 1, 1).expand(B, L, O, H).reshape(-1)
            age = (s_pos - sched["ev_pos"][p0]).clamp(min=1) if self.age_feature else None
        else:
            n = torch.zeros_like(p0)
            age = torch.ones_like(p0)
        if self.evid_gate and E:
            vals = vals * self._evid_gate(xf, vals, n, B, L)
        r = self._finish(vals, hit, n, age, sched.get("read_row"), longest_flag)
        return r.reshape(B, L, -1), beta.mean(-1)

    # Inference-time read ablations for locker5 (set ovl.read_mode; None = the trained read).
    # The read weights split into w0 (on the corpus row) and a distribution p over the local notes:
    #   flat_notes  p uniform over the valid notes (no context pairing), w0 as trained
    #   flat_prior  w0 = query-free Dirichlet weight e^b / (e^b + n), p as trained
    #   flat        both (a running mean of the notes, shrunk toward R_c by count only)
    #   last        p = the most recent note only (recency instead of pairing), w0 as trained
    #   prior_only  w0 = 1 (read R_c, ignore notes);  no_prior  w0 = 0 whenever a note exists
    READ_MODES = (None, "flat_notes", "flat_prior", "flat", "last", "prior_only", "no_prior")
    read_mode = None

    def _evid_gate(self, xf, D, n, B, L):
        """g [R, d_mem] for reads ordered (b, l, o, h); n = writes under the cue so far."""
        O, H, dm = len(self.orders), self.H, self.d_mem
        u = F.linear(xf, torch.cat([W.weight for W in self.W_g], 0)).view(B * L, O, 1, dm)
        u = u.expand(B * L, O, H, dm).reshape(-1, dm)  # [R, dm]
        stream = lambda p: p.view(1, O * H).expand(B * L, O * H).reshape(-1, 1)
        base = stream(self.g_count) * torch.log1p(n.to(xf.dtype)).unsqueeze(-1) + stream(self.g_bias)
        if self.evid_gate == "ctx":
            return torch.sigmoid(u + base)
        match = (u * D).sum(-1, keepdim=True) / math.sqrt(dm)
        return torch.sigmoid(stream(self.g_scale) * match + base)

    def _note_keys(self, xf, sched):
        """Key of each note: from the continuation's hidden state (+ the cue end's with ctx_key).
        A write at p has its cue ending at p - 1 >= 0 in the same row, so ev_flat - 1 stays in row."""
        keys = _proj_take(self.W_kn, xf, sched["ev_order"], sched["ev_flat"])
        if self.ctx_key:
            keys = keys + _proj_take(self.W_kc, xf, sched["ev_order"], sched["ev_flat"] - 1)
        return keys

    def _read_merged(self, xf, v, beta, sched, p0, hit, B, L):
        """note_merge: two-way scored read between the corpus row and the slot's delta-rule state."""
        O, H, dm = len(self.orders), self.H, self.d_mem
        seg_start = sched["ev_seg_start"]
        keys = self._note_keys(xf, sched)  # [E, dm]
        st = segmented_scan(1.0 - beta.repeat(1, 2), beta.repeat(1, 2) * torch.cat([v, keys], -1), seg_start,
                            int(sched["max_seg_len"]))
        st = _take(st, p0)  # [R, 2 dm]
        D, Ks = st[:, :dm], st[:, dm:]
        n = (p0 - seg_start[p0] + 1).to(xf.dtype)
        q = F.linear(xf, torch.cat([W.weight for W in self.W_qn], 0)).view(B * L, O, 1, dm)
        q = q.expand(B * L, O, H, dm).reshape(-1, dm)
        row = sched["read_row"].reshape(-1)
        R = self._prior_rows_at(row).to(xf.dtype)
        Wpk = torch.stack([W.weight for W in self.W_pk])
        pk = torch.einsum("nohi,oji->nohj", R.view(B * L, O, H, dm), Wpk).reshape(-1, dm)
        bias = self.prior_bias.view(1, O * H).expand(B * L, O * H).reshape(-1)
        s0 = (pk * q).sum(-1) / math.sqrt(dm) + bias
        s0 = torch.where(row >= 0, s0, float("-inf"))
        s1 = (Ks * q).sum(-1) / math.sqrt(dm) + torch.log(n.clamp(min=1))
        s1 = torch.where(hit.squeeze(-1), s1, float("-inf"))
        w = torch.softmax(torch.stack([s0, s1], -1), -1).nan_to_num(0.0)
        return w[:, 0:1] * R + w[:, 1:2] * D

    def _read_notes(self, xf, v, sched, p0, hit, B, L):
        """Attention over the last K notes of the addressed locker (same virtual slot, before s).
        Reads are ordered (b, l, o, h); every gather is an index_select (fast backward)."""
        O, H, K, dm = len(self.orders), self.H, self.notes, self.d_mem
        seg_start = sched["ev_seg_start"]
        idx = p0.unsqueeze(-1) - torch.arange(K, device=xf.device)  # [R, K] candidate events
        ok = hit & (idx >= seg_start[p0].unsqueeze(-1))
        idx = idx.clamp(min=0)
        keys = self._note_keys(xf, sched)  # [E, dm]
        q = F.linear(xf, torch.cat([W.weight for W in self.W_qn], 0)).view(B * L, O, 1, dm)
        q = q.expand(B * L, O, H, dm).reshape(-1, dm)  # [R, dm]
        scores = (_take(keys, idx) * q.unsqueeze(1)).sum(-1) / math.sqrt(dm)  # [R, K]
        scores = scores.masked_fill(~ok, float("-inf"))
        vals = _take(v, idx)
        mode = self.read_mode
        assert mode in self.READ_MODES, mode
        if not self.prior_note:
            assert mode is None, "read ablations are defined for locker5 (prior_note)"
            w = torch.softmax(scores, -1).nan_to_num(0.0)
            return (w.unsqueeze(-1) * vals).sum(1)
        row = sched["read_row"].reshape(-1)
        R = self._prior_rows_at(row).to(xf.dtype)  # [R, dm]
        Wpk = torch.stack([W.weight for W in self.W_pk])  # [O, dm, dm]
        pk = torch.einsum("nohi,oji->nohj", R.view(B * L, O, H, dm), Wpk).reshape(-1, dm)
        bias = self.prior_bias.view(1, O * H).expand(B * L, O * H).reshape(-1)  # [R]: reads are (b, l, o, h)
        s0 = (pk * q).sum(-1) / math.sqrt(dm) + bias
        s0 = torch.where(row >= 0, s0, float("-inf"))
        w = torch.softmax(torch.cat([s0.unsqueeze(-1), scores], -1), -1).nan_to_num(0.0)
        if mode is None:
            return w[:, 0:1] * R + (w[:, 1:].unsqueeze(-1) * vals).sum(1)
        okf = ok.to(xf.dtype)
        n = okf.sum(-1, keepdim=True)
        w0, wn = w[:, 0:1], w[:, 1:]
        p = wn / wn.sum(-1, keepdim=True).clamp(min=1e-30)
        if mode in ("flat_notes", "flat"):
            p = okf / n.clamp(min=1)
        if mode == "last":
            p = torch.zeros_like(p)
            p[:, 0] = okf[:, 0]  # idx[:, 0] = p0 = the most recent note
        if mode in ("flat_prior", "flat"):
            eb = bias.exp().unsqueeze(-1)
            w0 = eb / (eb + n)
        if mode == "prior_only":
            w0 = torch.ones_like(w0)
        if mode == "no_prior":
            w0 = (n == 0).to(xf.dtype)
        valid = (row >= 0).unsqueeze(-1).to(xf.dtype)
        w0 = torch.where(n > 0, w0 * valid, valid)  # no notes: read R_c; no valid row: notes only
        return w0 * R + (1.0 - w0) * (p.unsqueeze(-1) * vals).sum(1)

    def forward(self, x, sched, ablate=False):
        """ablate may also be {"miss": mask[B, L]}: at masked positions every read is
        replaced by an EMPTY read (zeros, flag 1), as if the cache had missed."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = _compute(x)
            m, beta = self.read(x32, sched)
            if self.training and self.read_dropout > 0:
                drop = torch.rand(m.shape[:2], device=m.device) < self.read_dropout
                m = _force_miss(m, drop, self.d_mem, self.r_width)
            if isinstance(ablate, dict):
                m = _force_miss(m, ablate["miss"], self.d_mem, self.r_width)
                ablate = False
            y, alpha = self.gate(x32, m)
            if self.hit_only:
                flags = m.reshape(*m.shape[:2], -1, self.r_width)[..., self.d_mem]  # 1 = empty
                y = y * (flags < 0.5).any(-1, keepdim=True).to(y.dtype)
        stats = dict(alpha=alpha.detach(), beta=beta.detach(), y_norm=y.detach().norm(dim=-1))
        return _inject(x, y, ablate), stats


class FrozenTable(nn.Module):
    """ctrl_frozen: static learned table (Engram-style), no in-context writes."""

    def __init__(self, d_model=128, orders=(2,), H=2, d_mem=32, rows=1 << 16, streams=None):
        super().__init__()
        self.orders, self.H, self.d_mem, self.rows = tuple(orders), H, d_mem, rows
        # streams: indices into the schedule's (order, head) streams this table reads (default all),
        # e.g. 2-3-gram cues only while in-context memory uses 1-5-grams.
        self.streams = list(streams) if streams is not None else None
        O = len(self.orders) if self.streams is None else len(self.streams) // H
        self.table = nn.Parameter(torch.randn(O * H, rows, d_mem) * 0.02)
        self.gate = _GateInject(d_model, O * H * (d_mem + 1))

    def forward(self, x, sched, ablate: bool = False):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = _compute(x)
            B, L, _ = x.shape
            row = sched["read_row"].reshape(B, L, -1)  # [B, L, O*H]
            if self.streams is not None:
                row = row[..., self.streams]
            valid = (row >= 0).unsqueeze(-1)
            stream = torch.arange(row.shape[-1], device=x.device)
            vals = self.table[stream, row.clamp(min=0)].to(x32.dtype)  # [B, L, OH, d_mem]
            vals = torch.where(valid, vals, 0.0)
            m = torch.cat([vals, (~valid).to(x32.dtype)], -1).reshape(B, L, -1)
            if isinstance(ablate, dict):
                m = _force_miss(m, ablate["miss"], self.d_mem)
                ablate = False
            y, alpha = self.gate(x32, m)
        stats = dict(alpha=alpha.detach(), y_norm=y.detach().norm(dim=-1))
        return _inject(x, y, ablate), stats


@torch.no_grad()
def reference_read(ovl: Overlay, x: torch.Tensor, tokens, mode="growing", M=None, ways=4) -> torch.Tensor:
    """Sequential loop over positions; independent of schedule.py and scan.py.
    Returns m [B, L, O*H*(d_mem+1)]. Used only in tests."""
    import numpy as np

    tokens = np.asarray(tokens)
    B, L, D = x.shape
    O, H, dm = len(ovl.orders), ovl.H, ovl.d_mem
    addr = {}
    for oi, n in enumerate(ovl.orders):
        for h in range(H):
            addr[oi, h] = ngram_addresses(tokens, n, h)
    out = torch.zeros(B, L, O, H, dm + 1, dtype=x.dtype)
    for b in range(B):
        if mode == "growing":
            slots, counts = {}, {}
        else:
            caches = {(oi, h): {} for oi in range(O) for h in range(H)}  # set -> OrderedDict(tag -> state)
        for t in range(L):
            for oi in range(O):
                for h in range(H):
                    a, val = addr[oi, h]
                    # read: slot of N-gram ending at t, before the write at t
                    state = None
                    if val[b, t]:
                        if mode == "growing":
                            state = slots.get((oi, h, int(a[b, t])))
                        else:
                            st, tg, _ = set_and_tag(np.int64(a[b, t]), M, ways)
                            od = caches[oi, h].setdefault(int(st), OrderedDict())
                            if int(tg) in od:
                                od.move_to_end(int(tg))
                                state = od[int(tg)][0]
                    if state is None:
                        out[b, t, oi, h, dm] = 1.0
                    else:
                        out[b, t, oi, h, :dm] = state
                    # write: slot of N-gram ending at t-1, content from x_t
                    if t >= 1 and val[b, t - 1] and tokens[b, t] != PAD:
                        if mode == "growing":
                            key = (oi, h, int(a[b, t - 1]))
                            prev = slots.get(key, torch.zeros(dm, dtype=x.dtype))
                            counts[key] = counts.get(key, 0) + 1
                            v, beta = ovl.event_values(x[b, t].unsqueeze(0), torch.tensor([oi]),
                                                       torch.tensor([counts[key]]))
                            v, beta = v[0], beta[0]
                            slots[key] = (1 - beta) * prev + beta * v
                        else:
                            st, tg, _ = set_and_tag(np.int64(a[b, t - 1]), M, ways)
                            od = caches[oi, h].setdefault(int(st), OrderedDict())
                            tg = int(tg)
                            if tg in od:
                                prev, n = od[tg]
                                od.move_to_end(tg)
                            else:
                                if len(od) >= ways:
                                    od.popitem(last=False)  # evict LRU
                                prev, n = torch.zeros(dm, dtype=x.dtype), 0
                            v, beta = ovl.event_values(x[b, t].unsqueeze(0), torch.tensor([oi]), torch.tensor([n + 1]))
                            v, beta = v[0], beta[0]
                            od[tg] = ((1 - beta) * prev + beta * v, n + 1)
    return out.reshape(B, L, -1)
