"""Slice D: writable N-gram embedding (Overlay prior_init). Scan vs a brute-force loop, and the
three switch settings (slice D / static table / plain document memory)."""
import numpy as np
import pytest
import torch

from src.hashing import ngram_addresses
from src.overlay import Overlay
from src.schedule import build_schedule
from src.train import sched_to_torch

ORDERS, H, DM, ROWS = (1, 2, 3), 1, 6, 64


def _setup(seed=0, B=2, L=40, **kw):
    torch.manual_seed(seed)
    ovl = Overlay(d_model=16, orders=ORDERS, H=H, d_mem=DM, write_mode="delta", prior_rows=ROWS,
                  prior_init=True, **kw).double()
    tok = np.random.default_rng(seed).integers(2, 7, size=(B, L))
    s = sched_to_torch(build_schedule(tok, orders=ORDERS, H=H, table_rows=ROWS), "cpu")
    x = torch.randn(B, L, 16, dtype=torch.float64)
    return ovl, tok, s, x


def _vals(ovl, x, s):
    m, _ = ovl.read(x, s)
    B, L = x.shape[:2]
    return m.view(B, L, len(ORDERS), H, -1)[..., :DM]


def _brute(ovl, tok, s, x):
    """Slots keyed by (order, address): start at R_c, delta writes; read before the write at s."""
    B, L = tok.shape
    rr = s["read_row"].numpy()
    out = torch.zeros(B, L, len(ORDERS), H, DM, dtype=x.dtype)
    for b in range(B):
        for oi, n in enumerate(ORDERS):
            addr, valid = ngram_addresses(tok[b:b + 1], n, 0)
            slots = {}
            for t in range(L):
                if valid[0, t]:  # read at t: cue ending at t
                    a = int(addr[0, t])
                    out[b, t, oi, 0] = slots[a] if a in slots else ovl.prior_table[oi * H, int(rr[b, t, oi, 0])]
                if t >= 1 and valid[0, t - 1]:  # write at t: cue ending at t-1, value from x[t]
                    a = int(addr[0, t - 1])
                    v, beta = ovl.event_values(x[b, t].unsqueeze(0), torch.tensor([oi]), torch.tensor([1]))
                    prev = slots.get(a, ovl.prior_table[oi * H, int(rr[b, t - 1, oi, 0])])
                    slots[a] = prev + beta[0] * (v[0] - prev)
    return out


def test_slice_d_scan_matches_brute_force():
    ovl, tok, s, x = _setup()
    with torch.no_grad():
        got, ref = _vals(ovl, x, s), _brute(ovl, tok, s, x)
    assert torch.allclose(got, ref, atol=1e-10), (got - ref).abs().max()


def test_writes_off_reads_rows_only():
    ovl, tok, s, x = _setup(writes=False)
    with torch.no_grad():
        got = _vals(ovl, x, s)
    rr = s["read_row"]
    for oi in range(len(ORDERS)):
        exp = torch.where((rr[..., oi, 0] >= 0).unsqueeze(-1), ovl.prior_table[oi][rr[..., oi, 0].clamp(min=0)], 0.0)
        assert torch.allclose(got[..., oi, 0, :], exp)


def test_zero_rows_equal_plain_document_memory():
    ovl, tok, s, x = _setup(prior_learned=False)
    plain = Overlay(d_model=16, orders=ORDERS, H=H, d_mem=DM, write_mode="delta").double()
    plain.load_state_dict({k: v for k, v in ovl.state_dict().items() if k != "prior_table"})
    with torch.no_grad():
        assert torch.allclose(ovl(x, s)[0], plain(x, s)[0], atol=1e-12)
    assert not ovl.prior_table.requires_grad


def test_gradients_reach_rows_only_when_learned():
    ovl, tok, s, x = _setup()
    ovl(x, s)[0].sum().backward()
    assert ovl.prior_table.grad is not None and ovl.prior_table.grad.abs().sum() > 0


def _brute_add(ovl, tok, s, x):
    """Delta N-gram-style: read R_c + D_c; D_c starts at 0, delta writes ignore R_c."""
    B, L = tok.shape
    rr = s["read_row"].numpy()
    out = torch.zeros(B, L, len(ORDERS), H, DM, dtype=x.dtype)
    for b in range(B):
        for oi, n in enumerate(ORDERS):
            addr, valid = ngram_addresses(tok[b:b + 1], n, 0)
            patch = {}
            for t in range(L):
                if valid[0, t]:
                    a = int(addr[0, t])
                    out[b, t, oi, 0] = ovl.prior_table[oi * H, int(rr[b, t, oi, 0])] + patch.get(a, 0.0)
                if t >= 1 and valid[0, t - 1]:
                    a = int(addr[0, t - 1])
                    v, beta = ovl.event_values(x[b, t].unsqueeze(0), torch.tensor([oi]), torch.tensor([1]))
                    prev = patch.get(a, torch.zeros(DM, dtype=x.dtype))
                    patch[a] = prev + beta[0] * (v[0] - prev)
    return out


def test_prior_add_matches_brute_force_and_differs_from_init():
    torch.manual_seed(0)
    add = Overlay(d_model=16, orders=ORDERS, H=H, d_mem=DM, write_mode="delta", prior_rows=ROWS,
                  prior_add=True).double()
    ovl, tok, s, x = _setup()
    add.load_state_dict(ovl.state_dict())  # identical parameters: only the write error differs
    with torch.no_grad():
        got, ref = _vals(add, x, s), _brute_add(add, tok, s, x)
        assert torch.allclose(got, ref, atol=1e-10), (got - ref).abs().max()
        assert not torch.allclose(got, _vals(ovl, x, s))


def test_additive2_prior_split_is_the_sum_unpacked():
    """prior_split: same parameters as additive; read = [D_c, flag, R_c] where additive reads D_c + R_c."""
    torch.manual_seed(0)
    add = Overlay(d_model=16, orders=ORDERS, H=H, d_mem=DM, write_mode="delta", prior_rows=ROWS, prior_add=True).double()
    torch.manual_seed(0)
    spl = Overlay(d_model=16, orders=ORDERS, H=H, d_mem=DM, write_mode="delta", prior_rows=ROWS, prior_add=True,
                  prior_split=True).double()
    sd = {k: v for k, v in add.state_dict().items() if not k.startswith("gate.")}
    spl.load_state_dict(sd, strict=False)
    _, tok, s, x = _setup()
    with torch.no_grad():
        a = add.read(x, s)[0].view(2, 40, len(ORDERS), H, -1)
        b = spl.read(x, s)[0].view(2, 40, len(ORDERS), H, -1)
    assert spl.r_width == 2 * DM + 1 and b.shape[-1] == 2 * DM + 1
    assert torch.allclose(b[..., :DM] + b[..., DM + 1:], a[..., :DM], atol=1e-12)
    assert torch.equal(b[..., DM], a[..., DM])  # empty flag unchanged


def test_additive2_value_ctx_matches_brute_force():
    """value_ctx: v = W_v x[t] + W_vc x[t - 1]; otherwise the additive rule."""
    torch.manual_seed(1)
    ovl = Overlay(d_model=16, orders=ORDERS, H=H, d_mem=DM, write_mode="delta", prior_rows=ROWS, prior_add=True,
                  value_ctx=True).double()
    _, tok, s, x = _setup(seed=1)
    B, L = tok.shape
    rr = s["read_row"].numpy()
    ref = torch.zeros(B, L, len(ORDERS), H, DM, dtype=x.dtype)
    with torch.no_grad():
        for b in range(B):
            for oi, n in enumerate(ORDERS):
                addr, valid = ngram_addresses(tok[b:b + 1], n, 0)
                patch = {}
                for t in range(L):
                    if valid[0, t]:
                        a = int(addr[0, t])
                        ref[b, t, oi, 0] = ovl.prior_table[oi * H, int(rr[b, t, oi, 0])] + patch.get(a, 0.0)
                    if t >= 1 and valid[0, t - 1]:
                        a = int(addr[0, t - 1])
                        v = ovl.W_v[oi](x[b, t]) + ovl.W_vc[oi](x[b, t - 1])
                        beta = torch.sigmoid(ovl.w_beta[oi](x[b, t]))
                        prev = patch.get(a, torch.zeros(DM, dtype=x.dtype))
                        patch[a] = prev + beta * (v - prev)
        got = _vals(ovl, x, s)
    assert torch.allclose(got, ref, atol=1e-10), (got - ref).abs().max()


@pytest.mark.parametrize("mode", ["ctx", "match"])
def test_evid_gate_matches_brute_force(mode):
    """evid_gate: read R_c + g * D_c; g = sigmoid(W_g x + a log(1+n) + b) ("ctx") or
    sigmoid(c (W_g x . D_c) / sqrt(d) + a log(1+n) + b) ("match"); n = writes under the cue so far."""
    import math

    torch.manual_seed(4)
    ovl = Overlay(d_model=16, orders=ORDERS, H=H, d_mem=DM, write_mode="delta", prior_rows=ROWS, prior_add=True,
                  evid_gate=mode).double()
    with torch.no_grad():
        for p in ovl.W_g.parameters():
            p.mul_(20)
        ovl.g_count.uniform_(-1, 1)
        ovl.g_bias.uniform_(-1, 1)
        if mode == "match":
            ovl.g_scale.uniform_(1, 3)
    _, tok, s, x = _setup(seed=4)
    B, L = tok.shape
    rr = s["read_row"].numpy()
    ref = torch.zeros(B, L, len(ORDERS), H, DM, dtype=x.dtype)
    with torch.no_grad():
        for b in range(B):
            for oi, n in enumerate(ORDERS):
                addr, valid = ngram_addresses(tok[b:b + 1], n, 0)
                patch, cnt = {}, {}
                for t in range(L):
                    if valid[0, t]:
                        a = int(addr[0, t])
                        R = ovl.prior_table[oi * H, int(rr[b, t, oi, 0])]
                        if a in patch:
                            D, k = patch[a], cnt[a]
                            u = ovl.W_g[oi](x[b, t])
                            base = ovl.g_count[oi * H] * math.log1p(k) + ovl.g_bias[oi * H]
                            g = torch.sigmoid(u + base) if mode == "ctx" else \
                                torch.sigmoid(ovl.g_scale[oi * H] * (u @ D) / math.sqrt(DM) + base)
                            ref[b, t, oi, 0] = R + g * D
                        else:
                            ref[b, t, oi, 0] = R
                    if t >= 1 and valid[0, t - 1]:
                        a = int(addr[0, t - 1])
                        v, beta = ovl.event_values(x[b, t].unsqueeze(0), torch.tensor([oi]), torch.tensor([1]))
                        prev = patch.get(a, torch.zeros(DM, dtype=x.dtype))
                        patch[a] = prev + beta[0] * (v[0] - prev)
                        cnt[a] = cnt.get(a, 0) + 1
        got = _vals(ovl, x, s)
    assert torch.allclose(got, ref, atol=1e-10), (got - ref).abs().max()
