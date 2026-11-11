"""Streaming (autoregressive) decode for every CPU/GPU backbone plus locker.

    dec = model.decoder(batch=1)
    logits = dec.prefill(prefix)      # parallel forward over the prefix, then load the decode state
    logits = dec.step(next_token)     # one token per call, O(1) work per locker cue

Locker state at decode time, per locker module:
  * a host-side hash index per cue stream (sequence b, order o, head h): a Python dict
    N-gram address -> slot id (growing mode), or per cache set an LRU OrderedDict tag -> slot id
    (bounded mode, the same policy as schedule._lru_sim);
  * a device-side value store val [n_slots, d_mem] (plus the last K note keys/values with notes=K);
  * host metadata per slot: notes written so far (count) and the last write position (age).

Each step hashes the cues ending at t on the host, gathers the slots read at t, builds the read
vector with the same Overlay._backoff / Overlay._finish / gate code as the parallel path, then
writes x_t into the slots addressed by the cues ending at t-1. These are schedule.py's rules: the
read at t sees writes at p < t only; the write at p goes to the cue ending at p-1 with content x_p.

Backbones: attention keeps a KV cache (RoPE or learned positions, optional short conv on q, k, v);
GDN / GLA step through fla's recurrent Cache; "none" and locker-only blocks have no state.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from .hashing import MOD, PRIMES, SEEDS, pattern_offsets
from .models import CausalAttention, FlaMixer, apply_rope
from .overlay import Overlay, _compute
from .scan import segmented_scan
from .schedule import addresses, build_schedule
from .tasks import BOS, PAD

M64 = (1 << 64) - 1


def mix64_int(a: int) -> int:
    """schedule.mix64 on one Python int (splitmix64 finalizer)."""
    z = a & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


class CueHasher:
    """Rolling N-gram addresses, one token at a time; equal to hashing.ngram_addresses.
    Cue streams are ordered (order, head), order-major, as in the schedule's [.., O, H] layout."""

    def __init__(self, orders, H: int, B: int):
        self.specs = []
        for n in orders:
            offs = pattern_offsets(n)
            for h in range(H):
                p = PRIMES[len(offs) - 1][h]
                self.specs.append((offs, [pow(p, r, MOD) for r in range(len(offs))], SEEDS[len(offs) - 1][h]))
        self.span = max(s[0][-1] for s in self.specs)
        self.B = B
        self.reset()

    def reset(self):
        self.hist = [[] for _ in range(self.B)]
        self.t = 0  # position of the next token
        self.prev = [[None] * len(self.specs) for _ in range(self.B)]  # cues ending at t-1

    def _addr(self, hist, t, spec):
        offs, pows, seed = spec
        if t - offs[-1] < 1 or len(hist) <= offs[-1]:
            return None
        x = 0
        for r, d in enumerate(offs):
            tok = hist[-1 - d]
            if tok == PAD or tok == BOS:
                return None
            x += (tok + 1) * pows[r]
        return (x % MOD) ^ seed

    def push(self, toks):
        """Hash tokens at position t (one per sequence). Returns (read, write): read[b][j] = address
        of the cue ending at t (None if invalid); write[b][j] = address the write at t goes to
        (the cue ending at t-1; None if invalid or token t is PAD)."""
        t = self.t
        read, write = [], []
        for b, tok in enumerate(toks):
            tok = int(tok)
            hb = self.hist[b]
            hb.append(tok)
            if len(hb) > self.span + 1:
                del hb[0]
            r = [self._addr(hb, t, s) for s in self.specs]
            write.append([None if tok == PAD else a for a in self.prev[b]] if t >= 1 else [None] * len(r))
            read.append(r)
            self.prev[b] = r
        self.t = t + 1
        return read, write

    def prime(self, htok: np.ndarray):
        """Set the history after a parallel prefill of htok [B, P]."""
        B, P = htok.shape
        self.reset()
        k = min(P, self.span + 1)
        self.t = P - k
        for i in range(P - k, P):
            self.push(htok[:, i])


class LockerStream:
    """Decode-time state of one Overlay module (see the module docstring)."""

    def __init__(self, ovl: Overlay, B: int, device, dtype, mode="growing", M=None, ways=4):
        self.ovl, self.B, self.device, self.dtype = ovl, B, device, dtype
        if getattr(ovl, "value_ctx", False) or getattr(ovl, "prior_split", False) or getattr(ovl, "evid_gate", None):
            raise NotImplementedError("streaming decode does not support value_ctx / prior_split / evid_gate yet")
        self.O, self.H = len(ovl.orders), ovl.H
        self.OH, self.R = self.O * self.H, B * self.O * self.H
        self.dm, self.K = ovl.d_mem, ovl.notes
        self.mode, self.ways = mode, ways
        if mode == "bounded":
            self.n_sets = M // ways
            assert self.n_sets * ways == M, "M must be divisible by ways"
        else:
            assert mode == "growing", mode
        self.refresh_weights()
        self.reset()

    @torch.no_grad()
    def refresh_weights(self):
        """Concatenated projections (one matmul per step for all orders). Call after changing weights."""
        ovl, dt, dev = self.ovl, self.dtype, self.device
        self.Wv = torch.cat([W.weight for W in ovl.W_v]).to(dt)
        self.Wb = torch.cat([w.weight for w in ovl.w_beta]).to(dt)
        self.bb = torch.cat([w.bias for w in ovl.w_beta]).to(dt)
        cols, off = [], 0
        for w in ovl.w_beta:
            n = w.out_features
            cols.append(torch.full((self.dm,), off) if n == 1 else torch.arange(off, off + n))
            off += n
        self.beta_cols = torch.stack(cols).to(dev)  # [O, d_mem] column of each order's beta channel
        self.is_mean = torch.tensor([m == "mean" for m in ovl.write_modes], device=dev)
        if self.K:
            self.Wq = torch.cat([W.weight for W in ovl.W_qn]).to(dt)
            self.Wk = torch.cat([W.weight for W in ovl.W_kn]).to(dt)
        r = torch.arange(self.R, device=dev)
        self.r_b, self.r_o = r // self.OH, (r // self.H) % self.O

    def reset(self, cap: int = 1024):
        self.cap, self.n = cap, 0
        self.val = torch.zeros(cap, self.dm, device=self.device, dtype=self.dtype)
        if self.K:
            self.nk = torch.zeros(cap, self.K, self.dm, device=self.device, dtype=self.dtype)
            self.nv = torch.zeros_like(self.nk)
        self.cnt = np.zeros(cap, np.int64)
        self.last = np.zeros(cap, np.int64)
        self.index = [dict() for _ in range(self.R)]  # growing: addr -> slot; bounded: set -> LRU dict

    def _grow(self, need: int):
        if need <= self.cap:
            return
        cap = max(2 * self.cap, need)

        def ext(t):
            out = t.new_zeros((cap, *t.shape[1:]))
            out[: self.cap] = t
            return out

        self.val = ext(self.val)
        if self.K:
            self.nk, self.nv = ext(self.nk), ext(self.nv)
        self.cnt = np.concatenate([self.cnt, np.zeros(cap - self.cap, np.int64)])
        self.last = np.concatenate([self.last, np.zeros(cap - self.cap, np.int64)])
        self.cap = cap

    def _alloc(self) -> int:
        self._grow(self.n + 1)
        self.n += 1
        return self.n - 1

    def _set_tag(self, a):
        z = mix64_int(a)
        return z % self.n_sets, (z >> 32) & 0xFFFFFFFF

    def _lookup(self, r, a) -> int:
        if self.mode == "growing":
            return self.index[r].get(a, -1)
        st, tg = self._set_tag(a)
        od = self.index[r].get(st)
        if od is not None and tg in od:
            od.move_to_end(tg)  # a read hit refreshes recency (schedule._lru_sim)
            return od[tg]
        return -1

    def _write_slot(self, r, a):
        """Slot that the write to address a goes to, and whether it starts empty."""
        if self.mode == "growing":
            s = self.index[r].get(a)
            if s is None:
                s = self.index[r][a] = self._alloc()
                return s, True
            return s, False
        st, tg = self._set_tag(a)
        od = self.index[r].setdefault(st, OrderedDict())
        if tg in od:
            od.move_to_end(tg)
            return od[tg], False
        s = od.popitem(last=False)[1] if len(od) >= self.ways else self._alloc()  # evict LRU, reuse slot
        od[tg] = s
        return s, True

    @torch.no_grad()
    def step(self, x, raddr, waddr, t):
        """x [B, d] (compute dtype): the locker's input at position t. Returns the read m [B, O*H*r_width]."""
        ovl, R, OH = self.ovl, self.R, self.OH
        slots, ns, ages, rows = [-1] * R, [0] * R, [1] * R, [-1] * R
        prior = ovl.prior_rows
        for b in range(self.B):
            for j, a in enumerate(raddr[b]):
                if a is None:
                    continue
                r = b * OH + j
                if prior:
                    rows[r] = mix64_int(a) % prior
                s = self._lookup(r, a)
                if s >= 0:
                    slots[r], ns[r], ages[r] = s, int(self.cnt[s]), max(1, t - int(self.last[s]))
        ev_r, ev_s, ev_new, ev_c = [], [], [], []
        for b in range(self.B):
            for j, a in enumerate(waddr[b]):
                if a is None:
                    continue
                r = b * OH + j
                s, new = self._write_slot(r, a)
                c = 1 if new else int(self.cnt[s]) + 1
                self.cnt[s], self.last[s] = c, t
                ev_r.append(r), ev_s.append(s), ev_new.append(int(new)), ev_c.append(c)
        E = len(ev_r)
        pack = torch.tensor(slots + ns + ages + rows + ev_r + ev_s + ev_new + ev_c, dtype=torch.long,
                            device=self.device)
        slot, n, age, row = pack[:R], pack[R:2 * R], pack[2 * R:3 * R], pack[3 * R:4 * R]
        # Read (before this step's writes: the read at t sees writes p < t).
        hit, flag = ovl._backoff((slot >= 0).unsqueeze(-1), self.B)
        p0 = torch.where(hit.squeeze(-1), slot, -1).clamp(min=0)
        if self.K:
            q = (x @ self.Wq.T).view(self.B, self.O, self.dm)[self.r_b, self.r_o]  # [R, d_mem]
            ok = hit & (torch.arange(self.K, device=self.device) < n.unsqueeze(-1))
            scores = (self.nk[p0] * q.unsqueeze(1)).sum(-1) / math.sqrt(self.dm)
            w = torch.softmax(scores.masked_fill(~ok, float("-inf")), -1).nan_to_num(0.0)
            vals = (w.unsqueeze(-1) * self.nv[p0]).sum(1)
        else:
            vals = torch.where(hit, self.val[p0], 0.0)
        m = ovl._finish(vals, hit, n, age, row, flag).reshape(self.B, -1)
        if E:  # write x_t into the slots of the cues ending at t-1
            e = pack[4 * R:].view(4, E)
            e_r, e_s, e_new, e_c = e[0], e[1], e[2].bool().unsqueeze(-1), e[3]
            e_b, e_o = self.r_b[e_r], self.r_o[e_r]
            v = (x @ self.Wv.T).view(self.B, self.O, self.dm)[e_b, e_o]
            if ovl.fixed_beta is not None:
                beta = torch.full_like(v, ovl.fixed_beta)
            else:
                sig = torch.sigmoid(x @ self.Wb.T + self.bb)[:, self.beta_cols]  # [B, O, d_mem]
                beta = torch.where(self.is_mean[e_o].unsqueeze(-1), (1.0 / e_c.to(v.dtype)).unsqueeze(-1),
                                   sig[e_b, e_o])
            prev = torch.where(e_new, 0.0, self.val[e_s])
            self.val[e_s] = (1.0 - beta) * prev + beta * v
            if self.K:
                k = (x @ self.Wk.T).view(self.B, self.O, self.dm)[e_b, e_o]
                ek = e_new.unsqueeze(-1)
                old_k = torch.where(ek, 0.0, self.nk[e_s])
                old_v = torch.where(ek, 0.0, self.nv[e_s])
                self.nk[e_s] = torch.cat([k.unsqueeze(1), old_k[:, :-1]], 1)
                self.nv[e_s] = torch.cat([v.unsqueeze(1), old_v[:, :-1]], 1)
        return m

    @torch.no_grad()
    def load_parallel(self, x, sched: dict, addr: np.ndarray):
        """Growing mode: final slot states from the parallel scan over a prefix.
        x [B, P, d] (compute dtype): the locker's input; sched: numpy schedule of the prefix;
        addr [B, P, O, H]: cue addresses (schedule.addresses)."""
        assert self.mode == "growing"
        self.reset()
        ovl, dev = self.ovl, self.device
        seg = np.asarray(sched["ev_seg_start"])
        E = len(seg)
        if E == 0:
            return
        B, P, D = x.shape
        xf = x.reshape(B * P, D)
        ev_flat = torch.as_tensor(sched["ev_flat"], device=dev)
        ev_order = torch.as_tensor(sched["ev_order"], device=dev)
        seg_t = torch.as_tensor(seg, device=dev)
        count = torch.arange(E, device=dev) - seg_t + 1
        v, beta = ovl._event_values_gather(xf, ev_flat, ev_order, count)
        is_last = np.ones(E, bool)
        is_last[:-1] = seg[1:] != seg[:-1]
        li = np.nonzero(is_last)[0]
        S = len(li)
        li_t = torch.as_tensor(li, device=dev)
        self._grow(S)
        if self.K:
            keys = torch.stack([W(xf) for W in ovl.W_kn])[ev_order, ev_flat]
            for k in range(self.K):
                idx = li_t - k
                ok = (idx >= seg_t[li_t]).unsqueeze(-1)
                self.nk[:S, k] = torch.where(ok, keys[idx.clamp(min=0)], 0.0)
                self.nv[:S, k] = torch.where(ok, v[idx.clamp(min=0)], 0.0)
        else:
            states = segmented_scan(1.0 - beta, beta * v, seg_t, int(sched["max_seg_len"]))
            self.val[:S] = states[li_t]
        self.cnt[:S] = li - seg[li] + 1
        self.last[:S] = np.asarray(sched["ev_pos"])[li]
        eb, ep, eo, eh = (np.asarray(sched[k])[li] for k in ("ev_b", "ev_pos", "ev_order", "ev_h"))
        stream = (eb * self.O + eo) * self.H + eh  # events are grouped by stream (schedule._growing)
        a = addr[eb, ep - 1, eo, eh]
        bounds = np.searchsorted(stream, np.arange(self.R + 1))
        for r in range(self.R):
            lo, hi = bounds[r], bounds[r + 1]
            self.index[r] = dict(zip(a[lo:hi].tolist(), range(lo, hi)))
        self.n = S

    @torch.no_grad()
    def load_synthetic(self, htok: np.ndarray):
        """Timing-only state: the exact slot set of the prefix (growing mode), values random.
        Used when the parallel prefill does not fit in memory."""
        assert self.mode == "growing"
        self.reset()
        _, _, w_addr, w_valid = addresses(htok, self.ovl.orders, self.H)
        B, P = htok.shape
        n0 = 0
        for b in range(B):
            for o in range(self.O):
                for h in range(self.H):
                    r = (b * self.O + o) * self.H + h
                    pos = np.nonzero(w_valid[b, :, o, h])[0]
                    a = w_addr[b, pos, o, h]
                    uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
                    last = np.zeros(len(uniq), np.int64)
                    np.maximum.at(last, inv, pos)
                    S = len(uniq)
                    self._grow(n0 + S)
                    self.cnt[n0:n0 + S], self.last[n0:n0 + S] = cnt, last
                    self.index[r] = dict(zip(uniq.tolist(), range(n0, n0 + S)))
                    n0 += S
        self.n = n0
        self.val[:n0].normal_(0, 0.02)
        if self.K:
            self.nk[:n0].normal_(0, 0.02)
            self.nv[:n0].normal_(0, 0.02)

    def state_bytes(self) -> dict:
        it = self.val.element_size()
        per_slot = self.dm * it * (2 * self.K if self.K else 1)
        return dict(n_slots=self.n, values=self.n * per_slot,
                    # count (int32) + last write position (int32, only read with age_feature)
                    meta=self.n * (4 + 4 * int(self.ovl.age_feature)),
                    # what a device-side open-addressing index would hold: 8-byte key + 4-byte slot
                    # id per entry at load factor 0.5. The host dicts used here are larger (CPython).
                    index_nominal=self.n * 24,
                    allocated_device=self.cap * per_slot)


def _tensor_bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def _fla_cache_bytes(cache) -> int:
    try:
        return sum(_tensor_bytes(cache[i]) for i in range(len(cache)))
    except Exception:
        return _tensor_bytes(getattr(cache, "states", None))


class Decoder:
    """Streaming decoder for a Model (see the module docstring).

    scfg: the arm's schedule config (mode, M, ways, read); only "growing" and "bounded" with normal
    reads are streamable (ctrl_random draws its reads from the whole sequence).
    max_len: preallocate attention KV caches for this many tokens (else they grow by doubling).
    hash_map: canonical id map applied to tokens before hashing (text.canonical_map), if the arm
    hashes canonical ids."""

    def __init__(self, model, batch: int = 1, scfg: dict | None = None, max_len: int | None = None,
                 hash_map: np.ndarray | None = None):
        self.model, self.B, self.max_len = model, batch, max_len
        scfg = dict(scfg or {})
        assert scfg.get("read", "normal") == "normal", "random reads (ctrl_random) are not streamable"
        self.mode, self.M, self.ways = scfg.get("mode", "growing"), scfg.get("M"), scfg.get("ways", 4)
        self.hash_map = hash_map
        p = next(model.parameters())
        self.device = p.device
        self.cdtype = torch.float64 if p.dtype == torch.float64 else torch.float32
        mods = [m for m in [model.static, *model.overlays.values()] if m is not None]
        for m in model.overlays.values():
            assert isinstance(m, Overlay), "frozen-table overlays: use cfg['static'] instead"
            assert not (m.training and m.read_dropout > 0), "call model.eval() first"
        if mods:
            cue = (tuple(mods[0].orders), mods[0].H)
            assert all((tuple(m.orders), m.H) == cue for m in mods), "all lockers share one cue set"
            self.orders, self.H = cue
            self.hasher = CueHasher(self.orders, self.H, batch)
        else:
            self.orders, self.hasher = None, None
        self.lockers = {k: LockerStream(m, batch, self.device, self.cdtype, self.mode, self.M, self.ways)
                        for k, m in model.overlays.items()}
        for blk in model.blocks:
            if blk.mixer is not None and not isinstance(blk.mixer, (CausalAttention, FlaMixer)):
                raise NotImplementedError(f"no streaming step for {type(blk.mixer).__name__}")
            if isinstance(blk.mixer, CausalAttention) and blk.mixer.ntk_train_len:
                raise NotImplementedError("dynamic NTK RoPE depends on the full length; not streamable")
        self.reset()

    def reset(self):
        self.t = 0
        if self.hasher is not None:
            self.hasher.reset()
        for lk in self.lockers.values():
            lk.reset()
        self.kv = {}  # block i -> dict(k, v, conv tail)
        self.fla = {}  # block i -> fla Cache

    # -------------------------------------------------------------- helpers
    def _htok(self, tokens: np.ndarray) -> np.ndarray:
        return self.hash_map[tokens] if self.hash_map is not None else tokens

    def _table_rows(self):
        rows = {m.rows for m in [self.model.static] if m is not None}
        rows |= {m.prior_rows for m in self.model.overlays.values() if m.prior_rows}
        assert len(rows) <= 1, f"static and prior tables must share one row count, got {rows}"
        return rows.pop() if rows else None

    def _static_step(self, x, raddr):
        st = self.model.static
        R = [[-1 if a is None else mix64_int(a) % st.rows for a in ra] for ra in raddr]
        row = torch.tensor(R, dtype=torch.long, device=self.device).view(self.B, 1, len(self.orders), self.H)
        y, _ = st(x.unsqueeze(1), {"read_row": row})
        return y.squeeze(1)

    def _locker_step(self, key, x, raddr, waddr):
        ovl = self.model.overlays[key]
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = _compute(x)
            m = self.lockers[key].step(x32, raddr, waddr, self.t)
            y, _ = ovl.gate(x32, m)
            if ovl.hit_only:
                flags = m.reshape(self.B, -1, ovl.r_width)[..., ovl.d_mem]
                y = y * (flags < 0.5).any(-1, keepdim=True).to(y.dtype)
        return x + y.to(x.dtype)

    def _attn_qkv(self, mix: CausalAttention, h, i, pos0, prefill):
        """q, k, v [B, heads, L, Dh] for h [B, L, D] at positions pos0.., same as CausalAttention."""
        B, L, D = h.shape
        a = mix.qkv(h)
        if mix.conv is not None:
            k = mix.conv.kernel_size[0]
            if prefill:
                tail = F.pad(a, (0, 0, max(0, k - 1 - L), 0))[:, -(k - 1):] if k > 1 else a[:, :0]
                a = F.silu(mix.conv(a.transpose(1, 2))[..., :L].transpose(1, 2))
            else:
                tail = self.kv.get(i, {}).get("tail")
                if tail is None:  # empty history: the conv's zero padding
                    tail = a.new_zeros(B, k - 1, a.shape[-1])
                win = torch.cat([tail.to(a.dtype), a], 1)  # [B, k, 3D], oldest first
                tail = win[:, 1:]
                cd = torch.float64 if win.dtype == torch.float64 else torch.float32  # accumulate like the conv
                w = mix.conv.weight[:, 0, :].T.to(cd)  # [k, 3D]
                bias = mix.conv.bias.to(cd) if mix.conv.bias is not None else 0
                a = F.silu((win.to(cd) * w).sum(1, keepdim=True) + bias).to(a.dtype)
            self.kv.setdefault(i, {})["tail"] = tail
        q, k, v = (t.contiguous() for t in a.view(B, L, 3, mix.heads, D // mix.heads).permute(2, 0, 3, 1, 4))
        if mix.rope:
            q, k = apply_rope(q, mix.rope_base, pos0), apply_rope(k, mix.rope_base, pos0)
        return q, k, v

    def _kv_store(self, i, k, v, t0):
        """Write k, v [B, heads, n, Dh] at positions t0.. into block i's cache."""
        c = self.kv.setdefault(i, {})
        need = t0 + k.shape[2]
        if "k" not in c or c["k"].shape[2] < need or c["k"].dtype != k.dtype:
            cap = max(need, self.max_len or 0, 2 * c["k"].shape[2] if "k" in c else 0)
            for name, src in (("k", k), ("v", v)):
                new = src.new_zeros(src.shape[0], src.shape[1], cap, src.shape[3])
                if name in c:
                    new[:, :, :t0] = c[name][:, :, :t0].to(src.dtype)
                c[name] = new
        c["k"][:, :, t0:need] = k
        c["v"][:, :, t0:need] = v

    def _attn_prefill(self, mix, h, i):
        q, k, v = self._attn_qkv(mix, h, i, 0, True)
        if q.is_cuda and mix.force_flash:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            if q.dtype not in (torch.float16, torch.bfloat16):
                q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True).to(h.dtype)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        self._kv_store(i, k, v, 0)
        B, L, D = h.shape
        return mix.out(o.transpose(1, 2).reshape(B, L, D))

    def _attn_step(self, mix, h, i):
        B, D = h.shape
        q, k, v = self._attn_qkv(mix, h.unsqueeze(1), i, self.t, False)
        c = self.kv.get(i, {})
        if "k" in c and c["k"].dtype != k.dtype:
            q, k, v = q.to(c["k"].dtype), k.to(c["k"].dtype), v.to(c["k"].dtype)
        self._kv_store(i, k, v, self.t)
        c = self.kv[i]
        o = F.scaled_dot_product_attention(q, c["k"][:, :, :self.t + 1], c["v"][:, :, :self.t + 1])
        return mix.out(o.to(h.dtype).transpose(1, 2).reshape(B, D))

    def _fla_run(self, mix: FlaMixer, h, i):
        from fla.models.utils import Cache

        cache = self.fla.get(i)
        if cache is None:
            cache = Cache()
        outs = []
        for s in range(0, h.shape[1], mix.segment):
            o, _, cache = mix.layer(h[:, s:s + mix.segment], past_key_values=cache, use_cache=True)
            outs.append(o)
        self.fla[i] = cache
        return torch.cat(outs, 1) if len(outs) > 1 else outs[0]

    # -------------------------------------------------------------- API
    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor, return_all: bool = False):
        """Parallel forward over tokens [B, P] (P >= 1), then load every decode state.
        Returns logits at the last position [B, V] (or all positions [B, P, V])."""
        self.reset()
        model = self.model
        B, P = tokens.shape
        assert B == self.B
        tok_np = tokens.cpu().numpy().astype(np.int64)
        htok = self._htok(tok_np)
        sched = st = addr = None
        if self.orders is not None:
            sched = build_schedule(htok, orders=self.orders, H=self.H, mode=self.mode, M=self.M,
                                   ways=self.ways, table_rows=self._table_rows())
            from .train import sched_to_torch

            st = sched_to_torch(sched, self.device)
            if self.mode == "growing" and self.lockers:
                addr = addresses(htok, self.orders, self.H)[0]
        x = model.embed(tokens)
        if model.pos is not None:
            x = x + model.pos(torch.arange(P, device=tokens.device))
        if model.static is not None:
            x, _ = model.static(x, st)

        def apply(at, x):
            key = str(at)
            if key not in model.overlays:
                return x
            out, _ = model.overlays[key](x, st)
            with torch.autocast(device_type=x.device.type, enabled=False):
                if self.mode == "growing":
                    self.lockers[key].load_parallel(_compute(x), sched, addr)
                else:  # bounded: replay the LRU cache over the prefix (O(P) host loop)
                    self._replay(key, _compute(x), htok)
            return out

        x = apply(0, x)
        for i, blk in enumerate(model.blocks):
            if blk.mixer is not None:
                h = blk.norm1(x)
                if isinstance(blk.mixer, CausalAttention):
                    x = x + self._attn_prefill(blk.mixer, h, i)
                else:
                    x = x + self._fla_run(blk.mixer, h, i)
            x = x + blk.mlp(blk.norm2(x))
            x = apply(i + 1, x)
        if self.hasher is not None:
            self.hasher.prime(htok)
        self.t = P
        h = model.norm(x if return_all else x[:, -1])
        return model.head(h)

    def _replay(self, key, x, htok):
        lk = self.lockers[key]
        lk.reset()
        hs = CueHasher(self.orders, self.H, self.B)
        for t in range(htok.shape[1]):
            r, w = hs.push(htok[:, t])
            lk.step(x[:, t], r, w, t)

    @torch.no_grad()
    def prefill_synthetic(self, tokens: torch.Tensor):
        """Timing-only decode state for a prefix that does not fit a parallel forward: the exact
        locker slot set with random values, attention KV caches with random entries, and GDN/GLA
        states from running each layer (chunked, through fla's Cache) on random inputs."""
        self.reset()
        model = self.model
        B, P = tokens.shape
        htok = self._htok(tokens.cpu().numpy().astype(np.int64))
        for lk in self.lockers.values():
            lk.load_synthetic(htok)
        D = model.embed.weight.shape[1]
        dt = torch.bfloat16 if self.device.type == "cuda" else model.embed.weight.dtype
        for i, blk in enumerate(model.blocks):
            if isinstance(blk.mixer, CausalAttention):
                hd = D // blk.mixer.heads
                k = torch.randn(B, blk.mixer.heads, P, hd, device=self.device, dtype=dt)
                self._kv_store(i, k, torch.randn_like(k), 0)
                del k
                if blk.mixer.conv is not None:
                    self.kv[i]["tail"] = torch.zeros(B, blk.mixer.conv.kernel_size[0] - 1, 3 * D,
                                                     device=self.device, dtype=dt)
            elif blk.mixer is not None:
                for s in range(0, P, blk.mixer.segment):
                    n = min(blk.mixer.segment, P - s)
                    self._fla_run(blk.mixer, torch.randn(B, n, D, device=self.device, dtype=dt), i)
        if self.hasher is not None:
            self.hasher.prime(htok)
        self.t = P

    @torch.no_grad()
    def step(self, token) -> torch.Tensor:
        """One token per sequence (tensor [B] or list of ints) at position self.t. Returns logits [B, V]."""
        model = self.model
        if not isinstance(token, torch.Tensor):
            token = torch.as_tensor(token, device=self.device)
        token = token.view(self.B).to(self.device)
        raddr = waddr = None
        if self.hasher is not None:
            tl = token.tolist()
            ht = self.hash_map[tl].tolist() if self.hash_map is not None else tl
            raddr, waddr = self.hasher.push(ht)
        x = model.embed(token)
        if model.pos is not None:
            x = x + model.pos.weight[self.t]
        if model.static is not None:
            x = self._static_step(x, raddr)
        if "0" in self.lockers:
            x = self._locker_step("0", x, raddr, waddr)
        for i, blk in enumerate(model.blocks):
            if blk.mixer is not None:
                h = blk.norm1(x)
                if isinstance(blk.mixer, CausalAttention):
                    x = x + self._attn_step(blk.mixer, h, i)
                else:
                    x = x + self._fla_run(blk.mixer, h.unsqueeze(1), i)[:, 0]
            x = x + blk.mlp(blk.norm2(x))
            if str(i + 1) in self.lockers:
                x = self._locker_step(str(i + 1), x, raddr, waddr)
        self.t += 1
        return model.head(model.norm(x))

    def state_bytes(self) -> dict:
        """Decode state, in bytes, by component (KV cache counted at its used length)."""
        out = dict(kv=0, kv_allocated=0, recurrent=0, locker=0, locker_slots=0)
        for c in self.kv.values():
            if "k" in c:
                it = c["k"].element_size()
                out["kv"] += 2 * c["k"].shape[0] * c["k"].shape[1] * self.t * c["k"].shape[3] * it
                out["kv_allocated"] += 2 * c["k"].numel() * it
            if "tail" in c:
                out["recurrent"] += _tensor_bytes(c["tail"])
        for cache in self.fla.values():
            out["recurrent"] += _fla_cache_bytes(cache)
        for key, lk in self.lockers.items():
            sb = lk.state_bytes()
            out[f"locker@{key}"] = sb
            out["locker"] += sb["values"] + sb["meta"]
            out["locker_slots"] += sb["n_slots"]
        out["total"] = out["kv"] + out["recurrent"] + out["locker"]
        return out
