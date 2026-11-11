"""CPU precompute of write events and read pointers (plan.md §4.4).

Semantics (per sequence b, order index o, hash head h):
  * write at position p (p >= 1, token p not PAD) goes to the slot addressed by
    the N-gram ending at p-1, with content derived from x_p.
  * read at position s reads the slot addressed by the N-gram ending at s, and
    sees only writes with p < s.

Output (numpy, events sorted by (virtual slot, position)):
  ev_flat      [E]  b*L + p  (gather index into x.reshape(B*L, d))
  ev_order     [E]  order index o (selects W_v[o], w_beta[o])
  ev_b, ev_pos [E]
  ev_h         [E]  hash head h (lets streaming decode rebuild the slot table)
  ev_seg_start [E]  index of the first event of the same virtual slot
  read_ptr     [B, L, O, H]  sorted event index of the state to read, or -1
  read_valid   [B, L, O, H]  the N-gram ending at s has an address
  read_row     [B, L, O, H]  addr mod table_rows (frozen control), -1 if invalid
  n_slots      [B]  number of virtual slots written per sequence
  max_seg_len  int  longest segment (bounds scan depth)
"""
from __future__ import annotations

import numpy as np

from .hashing import ngram_addresses
from .tasks import PAD

try:  # optional acceleration for the bounded-mode simulation
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


def addresses(tokens: np.ndarray, orders, H: int):
    B, L = tokens.shape
    O = len(orders)
    addr = np.zeros((B, L, O, H), dtype=np.int64)
    valid = np.zeros((B, L, O, H), dtype=bool)
    for oi, n in enumerate(orders):
        for h in range(H):
            addr[..., oi, h], valid[..., oi, h] = ngram_addresses(tokens, n, h)
    w_addr = np.zeros_like(addr)
    w_valid = np.zeros_like(valid)
    w_addr[:, 1:] = addr[:, :-1]
    w_valid[:, 1:] = valid[:, :-1] & (tokens[:, 1:] != PAD)[:, :, None, None]
    return addr, valid, w_addr, w_valid


def _segments(seg_key: np.ndarray) -> tuple[np.ndarray, int]:
    E = len(seg_key)
    if E == 0:
        return np.zeros(0, dtype=np.int64), 0
    new = np.ones(E, dtype=bool)
    new[1:] = seg_key[1:] != seg_key[:-1]
    idx = np.arange(E)
    seg_start = np.maximum.accumulate(np.where(new, idx, 0))
    max_len = int((idx - seg_start).max()) + 1
    return seg_start, max_len


def _growing(addr, valid, w_addr, w_valid):
    B, L, O, H = addr.shape
    eb, ep, eo, eh = np.nonzero(w_valid)
    rb, rs, ro, rh = np.nonzero(valid)
    nR, nE = len(rb), len(eb)
    G = np.concatenate([(rb * O + ro) * H + rh, (eb * O + eo) * H + eh])
    A = np.concatenate([addr[rb, rs, ro, rh], w_addr[eb, ep, eo, eh]])
    P = np.concatenate([rs, ep])
    T = np.concatenate([np.zeros(nR, np.int8), np.ones(nE, np.int8)])  # read before write at same p
    order = np.lexsort((T, P, A, G))
    Gs, As = G[order], A[order]
    is_ev = T[order] == 1
    N = len(order)
    newgrp = np.ones(N, dtype=bool)
    newgrp[1:] = (Gs[1:] != Gs[:-1]) | (As[1:] != As[:-1])
    grp = np.cumsum(newgrp) - 1
    ev_rank = np.cumsum(is_ev) - 1
    last = np.maximum.accumulate(np.where(is_ev, ev_rank, -1)) if N else np.zeros(0, np.int64)
    ev_grp = grp[is_ev]
    ok = last >= 0
    ok[ok] = ev_grp[last[ok]] == grp[ok]
    ptr = np.where(ok, last, -1)

    read_ptr = np.full((B, L, O, H), -1, dtype=np.int64)
    rsel = order[~is_ev]
    read_ptr[rb[rsel], rs[rsel], ro[rsel], rh[rsel]] = ptr[~is_ev]

    esel = order[is_ev] - nR
    ev_b, ev_pos, ev_o, ev_h = eb[esel], ep[esel], eo[esel], eh[esel]
    seg_start, max_len = _segments(ev_grp)
    return ev_b, ev_pos, ev_o, ev_h, ev_grp, seg_start, max_len, read_ptr


@njit(cache=True)
def _lru_sim(r_valid, r_set, r_tag, w_valid, w_set, w_tag, n_sets, ways):
    N, L = r_valid.shape
    ev_stream = np.empty(N * L, np.int64)
    ev_pos = np.empty(N * L, np.int64)
    ev_vslot = np.empty(N * L, np.int64)
    read_ev = np.full((N, L), -1, np.int64)
    ne = 0
    nv = 0
    for s in range(N):
        tags = np.full((n_sets, ways), -1, np.int64)
        vslot = np.full((n_sets, ways), -1, np.int64)
        last = np.full((n_sets, ways), -1, np.int64)
        used = np.zeros((n_sets, ways), np.int64)
        clock = 0
        for t in range(L):
            if r_valid[s, t]:
                st = r_set[s, t]
                tg = r_tag[s, t]
                for w in range(ways):
                    if tags[st, w] == tg:
                        read_ev[s, t] = last[st, w]
                        clock += 1
                        used[st, w] = clock
                        break
            if w_valid[s, t]:
                st = w_set[s, t]
                tg = w_tag[s, t]
                hit = -1
                for w in range(ways):
                    if tags[st, w] == tg:
                        hit = w
                        break
                if hit < 0:
                    for w in range(ways):
                        if tags[st, w] == -1:
                            hit = w
                            break
                    if hit < 0:
                        hit = 0
                        for w in range(1, ways):
                            if used[st, w] < used[st, hit]:
                                hit = w
                    tags[st, hit] = tg
                    vslot[st, hit] = nv  # new occupancy epoch
                    nv += 1
                ev_stream[ne] = s
                ev_pos[ne] = t
                ev_vslot[ne] = vslot[st, hit]
                last[st, hit] = ne
                ne += 1
                clock += 1
                used[st, hit] = clock
    return ev_stream[:ne], ev_pos[:ne], ev_vslot[:ne], read_ev


def mix64(addr) -> np.ndarray:
    """splitmix64 finalizer (bijective on 64 bits). The spec's raw address is
    nearly linear in the tokens for small n (x never wraps mod 2^61-1), so its
    high bits are constant and its low bits are structured. DEVIATIONS.md D4."""
    z = np.asarray(addr).astype(np.uint64)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def set_and_tag(addr, M: int, ways: int):
    n_sets = M // ways
    assert n_sets * ways == M, "M must be divisible by ways"
    z = mix64(addr)
    return (z % np.uint64(n_sets)).astype(np.int64), ((z >> np.uint64(32)) & np.uint64(0xFFFFFFFF)).astype(np.int64), n_sets


def _bounded(addr, valid, w_addr, w_valid, M, ways):
    B, L, O, H = addr.shape
    flat = lambda x: np.ascontiguousarray(x.transpose(0, 2, 3, 1).reshape(B * O * H, L))
    r_set, r_tag, n_sets = set_and_tag(addr, M, ways)
    w_set, w_tag, _ = set_and_tag(w_addr, M, ways)
    ev_stream, ev_pos, ev_vslot, read_ev = _lru_sim(
        flat(valid), flat(r_set), flat(r_tag), flat(w_valid), flat(w_set), flat(w_tag), n_sets, ways
    )
    order = np.lexsort((ev_pos, ev_vslot))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    ev_stream, ev_pos, ev_vslot = ev_stream[order], ev_pos[order], ev_vslot[order]
    ev_b = ev_stream // (O * H)
    ev_o = (ev_stream // H) % O
    ev_h = ev_stream % H
    # (no events at all: every read misses; inv is empty)
    ptr = np.where(read_ev >= 0, inv[np.maximum(read_ev, 0)], -1) if len(inv) else np.full_like(read_ev, -1)
    read_ptr = ptr.reshape(B, O, H, L).transpose(0, 3, 1, 2).copy()
    seg_start, max_len = _segments(ev_vslot)
    return ev_b, ev_pos, ev_o, ev_h, ev_vslot, seg_start, max_len, read_ptr


def _random_reads(ev_b, ev_pos, ev_o, ev_h, valid, u):
    """ctrl_random: each valid read takes a uniformly random EARLIER event of the
    same (sequence, order, head) stream. u: uniform [B, L, O, H], fixed per batch."""
    B, L, O, H = valid.shape
    g = (ev_b * O + ev_o) * H + ev_h
    key = g * L + ev_pos
    srt = np.argsort(key, kind="stable")
    ks = key[srt]
    rb, rs, ro, rh = np.nonzero(valid)
    rg = (rb * O + ro) * H + rh
    lo = np.searchsorted(ks, rg * L, "left")
    hi = np.searchsorted(ks, rg * L + rs, "left")
    n = hi - lo
    j = lo + np.minimum((u[rb, rs, ro, rh] * n).astype(np.int64), np.maximum(n - 1, 0))
    read_ptr = np.full((B, L, O, H), -1, dtype=np.int64)
    read_ptr[rb, rs, ro, rh] = np.where(n > 0, srt[np.minimum(j, len(srt) - 1)] if len(srt) else -1, -1)
    return read_ptr


def build_schedule(
    tokens: np.ndarray,
    orders=(2,),
    H: int = 2,
    mode: str = "growing",
    M: int | None = None,
    ways: int = 4,
    read: str = "normal",
    seed: int = 0,
    table_rows: int | None = None,
    joint: bool = False,
) -> dict:
    tokens = np.asarray(tokens, dtype=np.int64)
    B, L = tokens.shape
    O = len(orders)
    addr, valid, w_addr, w_valid = addresses(tokens, orders, H)
    if mode == "growing":
        ev_b, ev_pos, ev_o, ev_h, seg_key, seg_start, max_len, read_ptr = _growing(addr, valid, w_addr, w_valid)
    elif mode == "bounded":
        ev_b, ev_pos, ev_o, ev_h, seg_key, seg_start, max_len, read_ptr = _bounded(
            addr, valid, w_addr, w_valid, M, ways
        )
    else:
        raise ValueError(mode)
    if read == "random":
        u = np.random.default_rng(seed).random((B, L, O, H))
        read_ptr = _random_reads(ev_b, ev_pos, ev_o, ev_h, valid, u)
    elif read != "normal":
        raise ValueError(read)

    new_seg = np.ones(len(seg_key), dtype=bool)
    new_seg[1:] = seg_key[1:] != seg_key[:-1]
    out = dict(
        ev_flat=ev_b * L + ev_pos,
        ev_order=ev_o,
        ev_b=ev_b,
        ev_pos=ev_pos,
        ev_h=ev_h,
        ev_seg_start=seg_start,
        max_seg_len=max_len,
        read_ptr=read_ptr,
        read_valid=valid,
        n_slots=np.bincount(ev_b[new_seg], minlength=B),
    )
    if table_rows is not None:
        out["read_row"] = np.where(valid, (mix64(addr) % np.uint64(table_rows)).astype(np.int64), -1)
    if joint:  # dense slot ids for the joint (unified) delta rule, src/hgdn.joint_gdn
        out["j_read"], out["j_write"], out["j_slots"] = joint_slots(addr[..., 0], valid[..., 0],
                                                                   w_addr[..., 0], w_valid[..., 0])
    return out


def joint_slots(addr, valid, w_addr, w_valid):
    """Global slot id per (sequence, cue stream, cue) for the read cue (ending at s) and the write
    cue (ending at s-1) of every position; -1 where invalid. addr, valid, w_addr, w_valid: [B, L, O]
    (hash head 0). Returns (j_read, j_write) [B, L, O] int64 and the number of slots."""
    B, L, O = addr.shape
    tag = (np.arange(B)[:, None, None] * O + np.arange(O)[None, None, :]).astype(np.uint64)
    def key(a):  # mix (sequence, stream) into the 64-bit address; collisions are negligible
        return mix64(a.astype(np.uint64) ^ (tag * np.uint64(0x9E3779B97F4A7C15)))
    kr, kw = key(addr), key(w_addr)
    allk = np.concatenate([kr[valid], kw[w_valid]])
    uniq, inv = np.unique(allk, return_inverse=True)
    j_read = np.full((B, L, O), -1, dtype=np.int64)
    j_write = np.full((B, L, O), -1, dtype=np.int64)
    n_r = int(valid.sum())
    j_read[valid] = inv[:n_r]
    j_write[w_valid] = inv[n_r:]
    return j_read, j_write, np.int64(len(uniq))
