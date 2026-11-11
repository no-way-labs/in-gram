"""locker5 (Overlay prior_note): attention over a cue's last K notes plus the corpus row as a note."""
import numpy as np
import torch

from src.hashing import ngram_addresses
from src.overlay import Overlay
from src.schedule import build_schedule
from src.train import sched_to_torch

ORDERS, H, DM, ROWS, K = (1, 2, 3), 1, 6, 64, 3


def _setup(seed=0, B=2, L=40):
    torch.manual_seed(seed)
    ovl = Overlay(d_model=16, orders=ORDERS, H=H, d_mem=DM, write_mode="delta", prior_rows=ROWS,
                  notes=K, prior_note=True).double()
    tok = np.random.default_rng(seed).integers(2, 7, size=(B, L))
    s = sched_to_torch(build_schedule(tok, orders=ORDERS, H=H, table_rows=ROWS), "cpu")
    x = torch.randn(B, L, 16, dtype=torch.float64)
    return ovl, tok, s, x


def _brute(ovl, tok, s, x, mode=None):
    """Per (order, address): list of (key, value) notes; read attends over prior + last K notes."""
    B, L = tok.shape
    rr = s["read_row"].numpy()
    out = torch.zeros(B, L, len(ORDERS), H, DM, dtype=x.dtype)
    for b in range(B):
        for oi, n in enumerate(ORDERS):
            addr, valid = ngram_addresses(tok[b:b + 1], n, 0)
            notes = {}
            for t in range(L):
                if valid[0, t]:
                    a = int(addr[0, t])
                    q = ovl.W_qn[oi](x[b, t])
                    R = ovl.prior_table[oi * H, int(rr[b, t, oi, 0])]
                    ks = [ovl.W_pk[oi](R)] + [k for k, _ in notes.get(a, [])[-K:]]
                    vs = [R] + [v for _, v in notes.get(a, [])[-K:]]
                    sc = torch.stack([kk @ q for kk in ks]) / DM ** 0.5
                    sc[0] = sc[0] + ovl.prior_bias[oi * H]
                    w = torch.softmax(sc, 0)
                    n = len(vs) - 1
                    if mode is not None and n:
                        w0 = w[0]
                        p = w[1:] / w[1:].sum()
                        if mode in ("flat_notes", "flat"):
                            p = torch.full((n,), 1.0 / n, dtype=x.dtype)
                        if mode == "last":  # notes[-K:] is oldest first
                            p = torch.zeros(n, dtype=x.dtype)
                            p[-1] = 1.0
                        if mode in ("flat_prior", "flat"):
                            eb = ovl.prior_bias[oi * H].exp()
                            w0 = eb / (eb + n)
                        w0 = {"prior_only": torch.tensor(1.0, dtype=x.dtype),
                              "no_prior": torch.tensor(0.0, dtype=x.dtype)}.get(mode, w0)
                        w = torch.cat([w0.view(1), (1 - w0) * p])
                    out[b, t, oi, 0] = (w.unsqueeze(-1) * torch.stack(vs)).sum(0)
                if t >= 1 and valid[0, t - 1]:
                    a = int(addr[0, t - 1])
                    notes.setdefault(a, []).append((ovl.W_kn[oi](x[b, t]), ovl.W_v[oi](x[b, t])))
    return out


def test_locker5_matches_brute_force():
    ovl, tok, s, x = _setup()
    with torch.no_grad():
        for p in [*ovl.W_qn.parameters(), *ovl.W_kn.parameters(), *ovl.W_pk.parameters()]:
            p.mul_(30)  # make the scores matter
        ovl.prior_bias.fill_(0.3)
        m, _ = ovl.read(x, s)
        got = m.view(*x.shape[:2], len(ORDERS), H, -1)[..., :DM]
        ref = _brute(ovl, tok, s, x)
    valid = s["read_row"].numpy() >= 0
    assert torch.allclose(got[torch.from_numpy(valid)], ref[torch.from_numpy(valid)], atol=1e-10)


def test_locker5_equal_scores_is_dirichlet():
    ovl, tok, s, x = _setup(1)
    with torch.no_grad():
        for W in [*ovl.W_qn, *ovl.W_pk]:
            W.weight.zero_()
        m, _ = ovl.read(x, s)
    # first read of a fresh cue: no notes, so the read is exactly R_c
    rr = s["read_row"]
    b, t, o = 0, 1, 0  # position 0 is BOS (no cue); t=1 is the first order-1 read
    R = ovl.prior_table[o * H, int(rr[b, t, o, 0])]
    assert torch.allclose(m.view(*x.shape[:2], len(ORDERS), H, -1)[b, t, o, 0, :DM], R)


def test_locker5_gradients_reach_prior():
    ovl, tok, s, x = _setup(2)
    m, _ = ovl.read(x, s)
    m.sum().backward()
    assert ovl.prior_table.grad.abs().sum() > 0
    assert ovl.prior_bias.grad.abs().sum() > 0
    assert all(W.weight.grad.abs().sum() > 0 for W in ovl.W_pk)


def test_locker5_read_ablations_match_brute_force():
    ovl, tok, s, x = _setup(seed=1)
    with torch.no_grad():
        for p in [*ovl.W_qn.parameters(), *ovl.W_kn.parameters(), *ovl.W_pk.parameters()]:
            p.mul_(30)
        ovl.prior_bias.fill_(0.3)
        valid = torch.from_numpy(s["read_row"].numpy() >= 0)
        for mode in Overlay.READ_MODES[1:]:
            ovl.read_mode = mode
            m, _ = ovl.read(x, s)
            got = m.view(*x.shape[:2], len(ORDERS), H, -1)[..., :DM]
            ref = _brute(ovl, tok, s, x, mode)
            assert torch.allclose(got[valid], ref[valid], atol=1e-10), mode
        ovl.read_mode = "flat"  # the ablation must change the read on this data
        flat = ovl.read(x, s)[0]
        ovl.read_mode = None
        assert not torch.allclose(flat, ovl.read(x, s)[0])


def test_note_merge_matches_brute_force():
    """note_merge: delta-rule running value D and key K_s per slot (same beta); two-way softmax
    between R_c (key W_pk R_c, + bias) and the state (key K_s, + log n)."""
    torch.manual_seed(2)
    ovl = Overlay(d_model=16, orders=ORDERS, H=H, d_mem=DM, write_mode="delta", prior_rows=ROWS,
                  notes=1, prior_note=True, note_merge=True).double()
    B, L = 2, 40
    tok = np.random.default_rng(2).integers(2, 7, size=(B, L))
    s = sched_to_torch(build_schedule(tok, orders=ORDERS, H=H, table_rows=ROWS), "cpu")
    x = torch.randn(B, L, 16, dtype=torch.float64)
    with torch.no_grad():
        for p in [*ovl.W_qn.parameters(), *ovl.W_kn.parameters(), *ovl.W_pk.parameters()]:
            p.mul_(30)
        ovl.prior_bias.fill_(0.3)
        m, _ = ovl.read(x, s)
        got = m.view(B, L, len(ORDERS), H, -1)[..., :DM]
        rr = s["read_row"].numpy()
        ref = torch.zeros_like(got)
        for b in range(B):
            for oi, n in enumerate(ORDERS):
                addr, valid = ngram_addresses(tok[b:b + 1], n, 0)
                st = {}
                for t in range(L):
                    if valid[0, t]:
                        a = int(addr[0, t])
                        q = ovl.W_qn[oi](x[b, t])
                        R = ovl.prior_table[oi * H, int(rr[b, t, oi, 0])]
                        s0 = ovl.W_pk[oi](R) @ q / DM ** 0.5 + ovl.prior_bias[oi * H]
                        if a in st:
                            D, Ks, cnt = st[a]
                            s1 = Ks @ q / DM ** 0.5 + np.log(cnt)
                            w = torch.softmax(torch.stack([s0, s1]), 0)
                            ref[b, t, oi, 0] = w[0] * R + w[1] * D
                        else:
                            ref[b, t, oi, 0] = R
                    if t >= 1 and valid[0, t - 1]:
                        a = int(addr[0, t - 1])
                        be = torch.sigmoid(ovl.w_beta[oi](x[b, t]))
                        v, k = ovl.W_v[oi](x[b, t]), ovl.W_kn[oi](x[b, t])
                        D, Ks, cnt = st.get(a, (torch.zeros(DM, dtype=x.dtype), torch.zeros(DM, dtype=x.dtype), 0))
                        st[a] = (D + be * (v - D), Ks + be * (k - Ks), cnt + 1)
    valid = torch.from_numpy(rr >= 0)
    assert torch.allclose(got[valid], ref[valid], atol=1e-10)


def test_ctx_key_matches_brute_force():
    """ctx_key: note key = W_kn x[t] + W_kc x[t - 1] (continuation + the cue end it was written under)."""
    torch.manual_seed(3)
    ovl = Overlay(d_model=16, orders=ORDERS, H=H, d_mem=DM, write_mode="delta", prior_rows=ROWS,
                  notes=K, prior_note=True, ctx_key=True).double()
    B, L = 2, 40
    tok = np.random.default_rng(3).integers(2, 7, size=(B, L))
    s = sched_to_torch(build_schedule(tok, orders=ORDERS, H=H, table_rows=ROWS), "cpu")
    x = torch.randn(B, L, 16, dtype=torch.float64)
    with torch.no_grad():
        for p in [*ovl.W_qn.parameters(), *ovl.W_kn.parameters(), *ovl.W_kc.parameters(), *ovl.W_pk.parameters()]:
            p.mul_(30)
        ovl.prior_bias.fill_(0.3)
        m, _ = ovl.read(x, s)
        got = m.view(B, L, len(ORDERS), H, -1)[..., :DM]
        # brute force with the combined key: fold W_kc x[t-1] into a shifted copy of W_kn's input
        rr = s["read_row"].numpy()
        ref = torch.zeros_like(got)
        for b in range(B):
            for oi, n in enumerate(ORDERS):
                addr, valid = ngram_addresses(tok[b:b + 1], n, 0)
                notes = {}
                for t in range(L):
                    if valid[0, t]:
                        a = int(addr[0, t])
                        q = ovl.W_qn[oi](x[b, t])
                        R = ovl.prior_table[oi * H, int(rr[b, t, oi, 0])]
                        ks = [ovl.W_pk[oi](R)] + [k for k, _ in notes.get(a, [])[-K:]]
                        vs = [R] + [v for _, v in notes.get(a, [])[-K:]]
                        sc = torch.stack([kk @ q for kk in ks]) / DM ** 0.5
                        sc[0] = sc[0] + ovl.prior_bias[oi * H]
                        ref[b, t, oi, 0] = (torch.softmax(sc, 0).unsqueeze(-1) * torch.stack(vs)).sum(0)
                    if t >= 1 and valid[0, t - 1]:
                        a = int(addr[0, t - 1])
                        k = ovl.W_kn[oi](x[b, t]) + ovl.W_kc[oi](x[b, t - 1])
                        notes.setdefault(a, []).append((k, ovl.W_v[oi](x[b, t])))
    valid = torch.from_numpy(rr >= 0)
    assert torch.allclose(got[valid], ref[valid], atol=1e-10)
