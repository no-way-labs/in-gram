"""Hashed Gated DeltaNet ("unified head"): a Gated DeltaNet layer whose key space is
[learned dense keys (+) hashed one-hot cue keys].

Gated DeltaNet per head (fla convention, S: [d_k, d_v]):
    S_t = exp(g_t) S_{t-1};  S_t += k_t (beta_t (v_t - S_t^T k_t))^T;  o_t = S_t^T q_t
With a one-hot key k = e_c (c = hashed cue), only column c changes:
    col_c <- (1 - beta_t) exp(g_t) col_c + beta_t v_t          ("delta" write)
and normalized vanilla linear attention with one-hot keys gives the running mean of the values
written under c ("mean" write). So a locker slot is a GDN column addressed by a fixed sparse key.

This layer keeps the dense GDN (learned q, k) and adds, per head and per cue stream, hashed
columns that share the head's v_t, beta_t and g_t. Write at position p goes to the cue ending at
p-1 with value v_p; read at s takes the cue ending at s and sees writes p < s (src/schedule.py).
Read: o_t = dense_t + sum_o c_{t,o} col_o(cue_t), c = sigmoid(W_c x_t + b_c) per head and stream.

couple=True (one sweep of a joint delta rule): the dense state is written with
v'_t = v_t - r_{t-1}, where r_{t-1} is the hashed read at t-1, i.e. the hashed memory's prediction
of v_t (exact except when cue_{t-2} == cue_{t-1}, where the read misses the write at t-1). The
dense state then stores what the hashed columns fail to predict.
"""
from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import segmented_scan


def naive_gdn(q, k, v, g, beta, initial_state=None):
    """Reference recurrence. q, k [B, T, H, K] (already normalized and scaled), v [B, T, H, V],
    g, beta [B, T, H]. Returns o [B, T, H, V], final state [B, H, K, V]."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    S = q.new_zeros(B, H, K, V) if initial_state is None else initial_state.to(q.dtype)
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt = k[:, t]
        err = v[:, t] - torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, beta[:, t, :, None] * err)
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    return torch.stack(out, 1), S


def _fla_chunk():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except Exception:  # noqa: BLE001
        return None
    return chunk_gated_delta_rule


def dense_gdn(q, k, v, g, beta, segment: int | None = None, force_naive: bool = False):
    """Dense GDN with normalized q, k (q already scaled). fla's chunk kernel on CUDA (in segments
    of `segment` tokens carrying the state), else the naive recurrence."""
    fn = None if force_naive or not q.is_cuda else _fla_chunk()
    if fn is None:
        dt = torch.float64 if q.dtype == torch.float64 else torch.float32
        o, _ = naive_gdn(q.to(dt), k.to(dt), v.to(dt), g.to(dt), beta.to(dt))
        return o.to(v.dtype)
    kw = {}
    params = inspect.signature(fn).parameters
    if "head_first" in params:
        kw["head_first"] = False
    if "scale" in params:
        kw["scale"] = 1.0  # q is pre-scaled
    T = q.shape[1]
    seg = T if (segment is None or T <= segment) else segment
    outs, state = [], None
    for i in range(0, T, seg):
        sl = slice(i, i + seg)
        o, state = fn(q[:, sl], k[:, sl], v[:, sl], g[:, sl].float(), beta[:, sl], initial_state=state,
                      output_final_state=T > seg, **kw)
        outs.append(o)
    return torch.cat(outs, 1)


class ShortConv(nn.Module):
    """Causal depthwise conv + SiLU over [B, T, C]."""

    def __init__(self, C, size=4):
        super().__init__()
        self.conv = nn.Conv1d(C, C, size, groups=C, padding=size - 1, bias=False)

    def forward(self, x):
        T = x.shape[1]
        return F.silu(self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))


def hashed_columns(v, beta, g, sched, write_modes, B, L, heads, dv):
    """Per event state of every hashed column, then the read at each position.

    v [B, L, heads, dv]; beta, g [B, L, heads]. write_modes: one of "mean" / "delta" per stream
    (stream = order index in the schedule; H = 1). Returns vals [B, L, heads, O, dv] (0 where the
    cue has no earlier write)."""
    O = len(write_modes)
    ptr = sched["read_ptr"]  # [B, L, O, 1]
    assert ptr.shape[-1] == 1, "hgdn uses H = 1 hash head per cue stream"
    ptr = ptr[..., 0].reshape(-1)  # [B*L*O]
    hit = ptr >= 0
    p0 = ptr.clamp(min=0)
    seg_start = sched["ev_seg_start"]
    E = seg_start.shape[0]
    out = v.new_zeros(B * L * O, heads, dv)
    if E == 0:
        return out.view(B, L, O, heads, dv).permute(0, 1, 3, 2, 4)
    ev_flat = sched["ev_flat"]
    ev_order = sched["ev_order"]
    idx = torch.arange(E, device=v.device)
    first = idx == seg_start
    count = idx - seg_start + 1
    modes = torch.tensor([m == "delta" for m in write_modes], device=v.device)
    is_delta = modes[ev_order]  # [E]
    cdt = torch.float64 if v.dtype == torch.float64 else torch.float32
    vf = v.reshape(B * L, heads, dv)
    G = torch.cumsum(g.to(cdt), 1).reshape(B * L, heads)  # inclusive log-decay
    s_flat = torch.arange(B * L, device=v.device).repeat_interleave(O)  # read position per row
    read_delta = modes.repeat(B * L)  # stream of each read row
    for h in range(heads):  # one head at a time bounds peak memory on long inputs
        ve = vf[ev_flat, h].to(cdt)  # [E, dv]
        # mean: running mean as a scan, r_j = (1 - 1/n_j) r_{j-1} + v_j / n_j (no cross-slot sums)
        inv = (1.0 / count).to(cdt).unsqueeze(-1)
        mean_state = segmented_scan(1.0 - inv, inv * ve, seg_start)
        state = mean_state
        if bool(is_delta.any()):
            Ge = G[ev_flat, h]
            Gprev = torch.cat([Ge[:1], Ge[:-1]])
            b_ = beta.reshape(B * L, heads)[ev_flat, h].to(cdt)
            a = torch.where(first, torch.zeros_like(b_), torch.exp(Ge - Gprev) * (1.0 - b_))
            delta_state = segmented_scan(a.unsqueeze(-1), b_.unsqueeze(-1) * ve, seg_start)
            state = torch.where(is_delta.unsqueeze(-1), delta_state, mean_state)
        val = state[p0]  # [B*L*O, dv]
        if bool(is_delta.any()):  # decay from the last write to the read position
            dec = torch.exp(G[s_flat, h] - G[ev_flat[p0], h])
            val = torch.where(read_delta.unsqueeze(-1), val * dec.unsqueeze(-1), val)
        out[:, h] = torch.where(hit.unsqueeze(-1), val, 0.0).to(out.dtype)
    return out.view(B, L, O, heads, dv).permute(0, 1, 3, 2, 4)


class HashGDNMixer(nn.Module):
    """Gated DeltaNet with hashed one-hot cue columns per head (see module docstring).
    hash cfg: orders (cue streams, schedule order), write_mode (str or list per stream),
    couple (bool), gate_bias (init of the read gate's bias)."""

    def __init__(self, d, heads, head_dim=64, hash_cfg: dict | None = None, conv_size=4):
        super().__init__()
        hc = dict(hash_cfg or {})
        self.heads, self.dk, self.dv = heads, head_dim, head_dim
        self.orders = tuple(hc.get("orders", ()))
        wm = hc.get("write_mode", "mean")
        self.write_modes = [wm] * len(self.orders) if isinstance(wm, str) else list(wm)
        assert len(self.write_modes) == len(self.orders)
        self.couple = bool(hc.get("couple", False))
        # joint=True: "delta" streams join the dense key in one delta rule (joint_gdn); "mean"
        # streams stay decoupled reads (hashed_columns).
        self.joint = bool(hc.get("joint", False))
        self.joint_idx = [i for i, m in enumerate(self.write_modes) if m == "delta"] if self.joint else []
        self.mean_idx = [i for i, m in enumerate(self.write_modes) if m == "mean"] if self.joint else []
        self.chunk = int(hc.get("chunk", 64))
        assert not (self.joint and self.couple), "joint already shares one error; couple is the one-sweep version"
        hd = heads * head_dim
        self.q_proj = nn.Linear(d, hd, bias=False)
        self.k_proj = nn.Linear(d, hd, bias=False)
        self.v_proj = nn.Linear(d, hd, bias=False)
        self.q_conv, self.k_conv, self.v_conv = ShortConv(hd, conv_size), ShortConv(hd, conv_size), ShortConv(hd, conv_size)
        self.b_proj = nn.Linear(d, heads)
        self.a_proj = nn.Linear(d, heads)
        # GDN decay parametrization: g = -exp(A_log) * softplus(a + dt_bias)
        A = torch.empty(heads).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.g_proj = nn.Linear(d, hd, bias=False)
        self.o_norm = nn.RMSNorm(head_dim)
        self.o_proj = nn.Linear(hd, d, bias=False)
        if self.orders:
            self.c_proj = nn.Linear(d, heads * len(self.orders))
            nn.init.zeros_(self.c_proj.weight)
            nn.init.constant_(self.c_proj.bias, float(hc.get("gate_bias", 0.0)))
        if self.joint_idx:  # key mass split: softmax over [dense, joint streams] per token and head
            self.kw_proj = nn.Linear(d, heads * (1 + len(self.joint_idx)))
            nn.init.zeros_(self.kw_proj.weight)
            nn.init.zeros_(self.kw_proj.bias)
        self.segment = 1 << 18
        self.force_naive = False  # tests: use the reference recurrence on GPU too

    def forward(self, x, sched=None, return_parts=False):
        B, L, _ = x.shape
        Hh, dk, dv = self.heads, self.dk, self.dv
        f = (lambda t: t.double()) if x.dtype == torch.float64 else (lambda t: t.float())
        q = F.normalize(f(self.q_conv(self.q_proj(x)).view(B, L, Hh, dk)), dim=-1) / math.sqrt(dk)
        k = F.normalize(f(self.k_conv(self.k_proj(x)).view(B, L, Hh, dk)), dim=-1)
        v = f(self.v_conv(self.v_proj(x)).view(B, L, Hh, dv))
        beta = torch.sigmoid(f(self.b_proj(x)))
        g = -f(self.A_log).exp() * F.softplus(f(self.a_proj(x)) + self.dt_bias)
        if self.joint:
            return self._forward_joint(x, sched, q, k, v, beta, g, f, return_parts)
        r = None
        if self.orders:
            if sched is None:
                raise ValueError("hgdn with hashed columns needs the schedule")
            vals = hashed_columns(v, beta, g, sched, self.write_modes, B, L, Hh, dv)  # [B, L, Hh, O, dv]
            c = torch.sigmoid(f(self.c_proj(x))).view(B, L, Hh, len(self.orders), 1)
            r = (c * vals).sum(3)  # [B, L, Hh, dv]
        v_dense = v
        if self.couple and r is not None:
            v_dense = v - F.pad(r, (0, 0, 0, 0, 1, 0))[:, :L]  # minus the hashed prediction r_{t-1}
        dt = torch.bfloat16 if (x.is_cuda and torch.is_autocast_enabled()) else q.dtype
        o = f(dense_gdn(q.to(dt), k.to(dt), v_dense.to(dt), g, beta.to(dt),
                        segment=None if self.training else self.segment, force_naive=self.force_naive))
        o_dense = o
        if r is not None:
            o = o + r
        o = self.o_norm(o.to(self.o_norm.weight.dtype)) * F.silu(f(self.g_proj(x)).view(B, L, Hh, dv))
        y = self.o_proj(o.reshape(B, L, Hh * dv).to(self.o_proj.weight.dtype))
        if return_parts:
            return y, dict(q=q, k=k, v=v, v_dense=v_dense, beta=beta, g=g, r=r, o_dense=o_dense)
        return y

    def _forward_joint(self, x, sched, q, k, v, beta, g, f, return_parts):
        B, L, _ = x.shape
        Hh, dv = self.heads, self.dv
        if sched is None or "j_write" not in sched:
            raise ValueError("joint hgdn needs the schedule built with joint=True")
        c = torch.sigmoid(f(self.c_proj(x))).view(B, L, Hh, len(self.orders))
        Oj = len(self.joint_idx)
        w = torch.softmax(f(self.kw_proj(x)).view(B, L, Hh, 1 + Oj), -1)
        jw = sched["j_write"][..., self.joint_idx]
        jr = sched["j_read"][..., self.joint_idx]
        bw = w[..., 1:].sqrt() * (jw >= 0).unsqueeze(2).to(w.dtype)  # invalid cue: no key mass
        k_scaled = k * w[..., :1].sqrt()
        rq = c[..., self.joint_idx]
        with torch.autocast(device_type=x.device.type, enabled=False):  # fp32 around the triangular solve
            o = joint_gdn(q, k_scaled, v, g, beta, bw, rq, jw, jr, sched["j_slots"], chunk=self.chunk)
        r = None
        if self.mean_idx:
            modes = ["mean"] * len(self.orders)  # reads of the mean streams only (delta ones unused)
            vals = hashed_columns(v, beta, g, sched, modes, B, L, Hh, dv)[:, :, :, self.mean_idx]
            r = (c[..., self.mean_idx].unsqueeze(-1) * vals).sum(3)
            o = o + r
        o_joint = o
        o = self.o_norm(o.to(self.o_norm.weight.dtype)) * F.silu(f(self.g_proj(x)).view(B, L, Hh, dv))
        y = self.o_proj(o.reshape(B, L, Hh * dv).to(self.o_proj.weight.dtype))
        if return_parts:
            return y, dict(q=q, k=k_scaled, v=v, beta=beta, g=g, bw=bw, rq=rq, r=r, o_joint=o_joint)
        return y


# ---------------------------------------------------------------------------------------------
# Joint delta rule: one DeltaNet over the key [a_t k_t (+) sum_o b_{t,o} e_{w(t,o)}], i.e. dense
# learned keys and hashed one-hot cue keys in one state, with one error term
#     u_t = beta_t (v_t - S_d^T k_t - sum_o b_{t,o} H[w(t,o)])
# shared by both blocks. The dense block decays by exp(g_t); the hashed block does not decay.
# Query: [q_t (+) sum_o r_{t,o} e_{rd(t,o)}]. Reads follow GDN (after the write at t).

def naive_joint(q, k, v, g, beta, bw, rq, j_write, j_read, n_slots):
    """Reference recurrence. q, k [B, T, H, K] (k already scaled by a_t), v [B, T, H, V],
    g, beta [B, T, H]; bw, rq [B, T, H, O] key / query weights of the hashed streams;
    j_write, j_read [B, T, O] global slot ids (-1 = none). Returns o [B, T, H, V]."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    S = q.new_zeros(B, H, K, V)
    tab = q.new_zeros(n_slots, H, V)
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        pred = torch.einsum("bhk,bhkv->bhv", k[:, t], S)
        for b in range(B):
            for o in range(j_write.shape[-1]):
                w = int(j_write[b, t, o])
                if w >= 0:
                    pred[b] = pred[b] + bw[b, t, :, o, None] * tab[w]
        u = beta[:, t, :, None] * (v[:, t] - pred)
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], u)
        tab = tab.clone()
        for b in range(B):
            for o in range(j_write.shape[-1]):
                w = int(j_write[b, t, o])
                if w >= 0:
                    tab[w] = tab[w] + bw[b, t, :, o, None] * u[b]
        ot = torch.einsum("bhk,bhkv->bhv", q[:, t], S)
        for b in range(B):
            for o in range(j_read.shape[-1]):
                r = int(j_read[b, t, o])
                if r >= 0:
                    ot[b] = ot[b] + rq[b, t, :, o, None] * tab[r]
        out.append(ot)
    return torch.stack(out, 1)


def _gather_slots(tab, ids, wts):
    """sum_o wts[..., o] * tab[ids[..., o]]. tab [N, H, V]; ids [B, C, O]; wts [B, C, H, O]
    -> [B, C, H, V]."""
    ok = ids >= 0
    rows = tab[ids.clamp(min=0)]  # [B, C, O, H, V]
    w = (wts * ok.unsqueeze(2).to(wts.dtype)).permute(0, 1, 3, 2).unsqueeze(-1)  # [B, C, O, H, 1]
    return (rows * w).sum(2)


def joint_gdn(q, k, v, g, beta, bw, rq, j_write, j_read, n_slots, chunk: int = 64):
    """Chunkwise joint delta rule; same result as naive_joint. Parallel inside each chunk; the
    dense state [B, H, K, V] and the hashed table [n_slots, H, V] pass between chunks."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    O = j_write.shape[-1]
    C = chunk
    pad = (-T) % C
    if pad:
        z = lambda x: F.pad(x, (0, 0) * (x.dim() - 2) + (0, pad))  # noqa: E731
        q, k, v, g, beta, bw, rq = (z(x) for x in (q, k, v, g, beta, bw, rq))
        j_write = F.pad(j_write, (0, 0, 0, pad), value=-1)
        j_read = F.pad(j_read, (0, 0, 0, pad), value=-1)
    N = (T + pad) // C
    dt = q.dtype
    # chunked views: [B, N, C, ...] -> head-major [B, H, N, C, ...]
    cq = q.view(B, N, C, H, K).permute(0, 3, 1, 2, 4)
    ck = k.view(B, N, C, H, K).permute(0, 3, 1, 2, 4)
    cv = v.view(B, N, C, H, V).permute(0, 3, 1, 2, 4)
    cb = beta.view(B, N, C, H).permute(0, 3, 1, 2)
    gam = g.view(B, N, C, H).permute(0, 3, 1, 2).cumsum(-1)  # in-chunk cumulative log decay
    jw = j_write.view(B, N, C, O)
    jr = j_read.view(B, N, C, O)
    bwc = bw.view(B, N, C, H, O)
    rqc = rq.view(B, N, C, H, O)
    tril_strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    tril_incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    Gam = torch.exp((gam.unsqueeze(-1) - gam.unsqueeze(-2)).masked_fill(~tril_incl, float("-inf")))  # [B,H,N,C,C]
    # exact-match kernels between cue slots inside a chunk (same stream o only: ids encode o)
    eq_kk = (jw.unsqueeze(3) == jw.unsqueeze(2)) & (jw.unsqueeze(3) >= 0)  # [B, N, Ci, Cj, O]
    eq_qk = (jr.unsqueeze(3) == jw.unsqueeze(2)) & (jr.unsqueeze(3) >= 0)
    bwh = bwc.permute(0, 3, 1, 2, 4)  # [B, H, N, C, O]
    rqh = rqc.permute(0, 3, 1, 2, 4)
    M_kk = torch.einsum("bhnio,bhnjo,bnijo->bhnij", bwh, bwh, eq_kk.to(dt))
    M_qk = torch.einsum("bhnio,bhnjo,bnijo->bhnij", rqh, bwh, eq_qk.to(dt))
    A = (ck @ ck.transpose(-1, -2)) * Gam + M_kk
    Lmat = (cb.unsqueeze(-1) * A).masked_fill(~tril_strict, 0.0)
    eye = torch.eye(C, dtype=dt, device=q.device)
    Tinv = torch.linalg.solve_triangular(eye + Lmat, eye.expand_as(Lmat), upper=False, unitriangular=True)
    Aqk = ((cq @ ck.transpose(-1, -2)) * Gam + M_qk).masked_fill(~tril_incl, 0.0)
    eg = gam.exp().unsqueeze(-1)  # [B, H, N, C, 1]
    decay_out = torch.exp(gam[..., -1:] - gam).unsqueeze(-1)  # e^{gamma_C - gamma_j}
    S = q.new_zeros(B, H, K, V)
    tab = q.new_zeros(int(n_slots), H, V)
    outs = []
    grad = torch.is_grad_enabled()
    for n in range(N):
        Rk = _gather_slots(tab, jw[:, n], bwc[:, n]).permute(0, 2, 1, 3)  # [B, H, C, V]
        Rq = _gather_slots(tab, jr[:, n], rqc[:, n]).permute(0, 2, 1, 3)
        rhs = cb[:, :, n].unsqueeze(-1) * (cv[:, :, n] - eg[:, :, n] * (ck[:, :, n] @ S) - Rk)
        u = Tinv[:, :, n] @ rhs  # [B, H, C, V]
        o = eg[:, :, n] * (cq[:, :, n] @ S) + Rq + Aqk[:, :, n] @ u
        outs.append(o)
        S = gam[:, :, n, -1].exp()[..., None, None] * S + (ck[:, :, n] * decay_out[:, :, n]).transpose(-1, -2) @ u
        # hashed table: index_add of b_{j,o} u_j into slot w(j, o)
        ids = jw[:, n].reshape(-1)  # [B*C*O] (b, c, o order)
        src = (bwc[:, n].permute(0, 1, 3, 2).unsqueeze(-1) * u.permute(0, 2, 1, 3).unsqueeze(2))  # [B, C, O, H, V]
        ok = ids >= 0
        src = src.reshape(-1, H, V)[ok]
        if grad:
            tab = tab.index_add(0, ids[ok], src)
        else:
            tab.index_add_(0, ids[ok], src)
    o = torch.stack(outs, 2).permute(0, 2, 3, 1, 4).reshape(B, N * C, H, V)
    return o[:, :T]
