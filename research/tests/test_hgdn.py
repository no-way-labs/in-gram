"""Hashed Gated DeltaNet (src/hgdn.py): hashed columns vs brute force, the one-hot-key GDN identity,
coupling, causality, the hash-off path, and (GPU only) fla's chunk kernel vs the reference."""
import numpy as np
import pytest
import torch

from src.controls import arm_config
from src.hgdn import HashGDNMixer, dense_gdn, hashed_columns, naive_gdn
from src.models import Model
from src.schedule import build_schedule
from src.train import sched_to_torch



@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _tokens(B, L, vocab, seed, no_adjacent_repeats=False):
    rng = np.random.default_rng(seed)
    t = rng.integers(2, 2 + vocab, size=(B, L))
    if no_adjacent_repeats:
        for b in range(B):
            for i in range(1, L):
                while t[b, i] == t[b, i - 1]:
                    t[b, i] = rng.integers(2, 2 + vocab)
    return t


def _brute(tokens, orders, modes, v, beta, g):
    """Dict-of-columns reference: write at p to cue ending at p-1 (value v_p); read at s from the
    cue ending at s, seeing writes p < s. Delta columns decay by exp(g) every step."""
    B, L = tokens.shape
    heads, dv = v.shape[2], v.shape[3]
    out = torch.zeros(B, L, heads, len(orders), dv)
    for b in range(B):
        for oi, (n, mode) in enumerate(zip(orders, modes)):
            cols = {}  # cue -> (state [heads, dv], count)
            for s in range(L):
                # decay every delta column to time s (GDN applies exp(g_s) at step s)
                if mode == "delta":
                    for c in cols:
                        st, cnt = cols[c]
                        cols[c] = (st * g[b, s].exp()[:, None], cnt)
                # write at p = s: cue ending at s-1 (needs s-1 >= n-1)
                if s - n >= 1:  # cues never include position 0 (BOS), as in hashing.ngram_addresses
                    cue = tuple(tokens[b, s - n:s])
                    st, cnt = cols.get(cue, (torch.zeros(heads, dv), 0))
                    if mode == "mean":
                        st = st + (v[b, s] - st) / (cnt + 1)
                    else:
                        st = st + beta[b, s][:, None] * (v[b, s] - st)
                    new = (st, cnt + 1)
                else:
                    new = None
                # read at s from cue ending at s, before the write at s (write p < s only)
                if s - n + 1 >= 1:
                    cue_r = tuple(tokens[b, s - n + 1:s + 1])
                    if cue_r in cols:
                        out[b, s, :, oi] = cols[cue_r][0]
                if new is not None:
                    cols[tuple(tokens[b, s - n:s])] = new
    return out


@pytest.mark.parametrize("modes", [("mean", "mean"), ("delta", "delta"), ("mean", "delta")])
def test_hashed_columns_match_brute_force(modes):
    B, L, heads, dv, orders = 2, 40, 2, 3, (1, 2)
    tok = _tokens(B, L, 4, 0)
    sched = sched_to_torch(build_schedule(tok, orders=orders, H=1), "cpu")
    g_ = torch.Generator().manual_seed(0)
    v = torch.randn(B, L, heads, dv, generator=g_)
    beta = torch.rand(B, L, heads, generator=g_)
    g = -torch.rand(B, L, heads, generator=g_) * 0.3
    got = hashed_columns(v, beta, g, sched, list(modes), B, L, heads, dv)  # [B, L, heads, O, dv]
    ref = _brute(tok, orders, modes, v, beta, g)
    assert torch.allclose(got, ref, atol=1e-10), (got - ref).abs().max()


def test_delta_columns_equal_gdn_with_one_hot_keys():
    """Locker's delta write is a Gated DeltaNet head with one-hot cue keys (paper claim)."""
    B, L, heads, dv, n = 1, 50, 2, 3, 2
    tok = _tokens(B, L, 5, 1, no_adjacent_repeats=True)
    sched = sched_to_torch(build_schedule(tok, orders=(n,), H=1), "cpu")
    g_ = torch.Generator().manual_seed(1)
    v = torch.randn(B, L, heads, dv, generator=g_)
    beta = torch.rand(B, L, heads, generator=g_)
    g = -torch.rand(B, L, heads, generator=g_) * 0.3
    got = hashed_columns(v, beta, g, sched, ["delta"], B, L, heads, dv)[:, :, :, 0]
    cues = sorted({tuple(tok[0, i - n + 1:i + 1]) for i in range(n, L)})
    ix = {c: i for i, c in enumerate(cues)}
    K = len(cues)
    k = torch.zeros(B, L, heads, K)
    q = torch.zeros(B, L, heads, K)
    bt = beta.clone()
    for t in range(L):
        if t - n >= 1:  # write key: cue ending at t-1 (cues exclude position 0, BOS)
            k[0, t, :, ix[tuple(tok[0, t - n:t])]] = 1.0
        else:
            bt[0, t] = 0.0  # no write
        if t - n + 1 >= 1:
            q[0, t, :, ix[tuple(tok[0, t - n + 1:t + 1])]] = 1.0
    o, _ = naive_gdn(q, k, v, g, bt)
    # GDN reads after the write at t; the schedule reads before it. They agree when the write key
    # at t differs from the read key at t, which no_adjacent_repeats guarantees for n=2.
    assert torch.allclose(got, o, atol=1e-10), (got - o).abs().max()


def _mixer(couple, modes=("mean", "delta"), orders=(1, 2)):
    torch.manual_seed(0)
    m = HashGDNMixer(16, 2, head_dim=8, hash_cfg=dict(orders=orders, write_mode=list(modes), couple=couple,
                                                     gate_bias=0.5))
    return m.double()


def test_coupling_subtracts_previous_read():
    B, L = 2, 24
    tok = _tokens(B, L, 4, 2)
    sched = sched_to_torch(build_schedule(tok, orders=(1, 2), H=1), "cpu")
    m = _mixer(True)
    x = torch.randn(B, L, 16)
    _, p = m(x, sched, return_parts=True)
    shifted = torch.cat([torch.zeros_like(p["r"][:, :1]), p["r"][:, :-1]], 1)
    assert torch.allclose(p["v_dense"], p["v"] - shifted)
    assert p["r"].abs().sum() > 0


@pytest.mark.parametrize("couple", [False, True])
def test_model_causal(couple):
    mcfg, scfg = arm_config("hgdn", dict(name="lm", L_max=32), {"d_model": 16, "heads": 2, "n_blocks": 2,
                                                                 "hash.couple": couple,
                                                                 "hash.orders": [1, 2, 3],
                                                                 "hash.write_mode": ["mean", "delta", "mean"]})
    mcfg["vocab"] = 20
    model = Model(mcfg).double().eval()
    for blk in model.blocks:
        blk.mixer.force_naive = True
    tok = _tokens(1, 32, 6, 3)
    t2 = tok.copy()
    t2[0, 20:] = _tokens(1, 12, 6, 4)
    outs = []
    for t in (tok, t2):
        s = sched_to_torch(build_schedule(t, **scfg), "cpu")
        outs.append(model(torch.as_tensor(t), s)[0])
    assert torch.allclose(outs[0][:, :20], outs[1][:, :20], atol=1e-10)
    assert not torch.allclose(outs[0][:, 20:], outs[1][:, 20:])


def test_hash_off_is_dense_gdn():
    m = HashGDNMixer(16, 2, head_dim=8, hash_cfg=dict(orders=())).double()
    x = torch.randn(2, 12, 16)
    y, p = m(x, None, return_parts=True)
    assert p["r"] is None and torch.equal(p["v_dense"], p["v"])
    mcfg, scfg = arm_config("hgdn", dict(name="lm", L_max=16), {"hash.orders": []})
    assert scfg["orders"] == () and mcfg["hash"]["orders"] == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fla chunk kernel needs CUDA")
def test_fla_chunk_matches_naive_gpu():
    torch.set_default_dtype(torch.float32)  # restored by the fixture
    if True:
        dev = "cuda"
        B, T, H, K = 2, 300, 2, 64
        g_ = torch.Generator(device=dev).manual_seed(0)
        q = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=g_), dim=-1) / K ** 0.5
        k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, generator=g_), dim=-1)
        v = torch.randn(B, T, H, K, device=dev, generator=g_)
        beta = torch.rand(B, T, H, device=dev, generator=g_)
        g = -torch.rand(B, T, H, device=dev, generator=g_) * 0.2
        ref, _ = naive_gdn(q, k, v, g, beta)
        got = dense_gdn(q, k, v, g, beta)
        assert (got.float() - ref).abs().max() < 2e-3 * ref.abs().max()
        seg = dense_gdn(q, k, v, g, beta, segment=128)  # segmented state carry
        assert (seg.float() - ref).abs().max() < 2e-3 * ref.abs().max()


# ---- joint delta rule ------------------------------------------------------------------------
from src.hgdn import joint_gdn, naive_joint  # noqa: E402


def _joint_inputs(B=2, T=50, H=2, K=4, V=3, O=3, vocab=4, seed=0):
    tok = _tokens(B, T, vocab, seed)
    s = build_schedule(tok, orders=tuple(range(1, O + 1)), H=1, joint=True)
    g_ = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g_), dim=-1) / 2
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g_), dim=-1)
    w = torch.softmax(torch.randn(B, T, H, 1 + O, generator=g_), -1)
    jw, jr = torch.as_tensor(s["j_write"]), torch.as_tensor(s["j_read"])
    bw = w[..., 1:].sqrt() * (jw >= 0).unsqueeze(2)
    k = k * w[..., :1].sqrt()
    v = torch.randn(B, T, H, V, generator=g_)
    g = -torch.rand(B, T, H, generator=g_) * 0.5
    beta = torch.rand(B, T, H, generator=g_)
    rq = torch.rand(B, T, H, O, generator=g_)
    return q, k, v, g, beta, bw, rq, jw, jr, int(s["j_slots"]), tok, s


@pytest.mark.parametrize("chunk", [8, 16, 64])
def test_joint_chunked_matches_naive(chunk):
    q, k, v, g, beta, bw, rq, jw, jr, n, *_ = _joint_inputs()
    ref = naive_joint(q, k, v, g, beta, bw, rq, jw, jr, n)
    got = joint_gdn(q, k, v, g, beta, bw, rq, jw, jr, n, chunk=chunk)
    assert torch.allclose(got, ref, atol=1e-10), (got - ref).abs().max()


def test_joint_without_hashed_mass_is_gdn():
    q, k, v, g, beta, bw, rq, jw, jr, n, *_ = _joint_inputs()
    got = joint_gdn(q, k, v, g, beta, bw * 0, rq * 0, jw, jr, n, chunk=16)
    ref, _ = naive_gdn(q, k, v, g, beta)
    assert torch.allclose(got, ref, atol=1e-10)


def test_joint_slots_consistent():
    *_, tok, s = _joint_inputs(B=2, T=30)
    jr, jw = s["j_read"], s["j_write"]
    for o, n in enumerate((1, 2, 3)):
        for b in range(2):
            for t in range(1, 30):
                if jw[b, t, o] >= 0:  # write cue at t is the read cue at t-1
                    assert jw[b, t, o] == jr[b, t - 1, o]
                    assert tuple(tok[b, t - n:t]) == tuple(tok[b, t - 1 - n + 1:t])
    # same cue, different sequences or streams -> different slots
    ids_b0 = set(jr[0][jr[0] >= 0].ravel()); ids_b1 = set(jr[1][jr[1] >= 0].ravel())
    assert not ids_b0 & ids_b1


@pytest.mark.parametrize("modes", [["delta"] * 3, ["mean", "mean", "delta"]])
def test_joint_model_causal_and_grads(modes):
    mcfg, scfg = arm_config("hgdn", dict(name="lm", L_max=40), {"d_model": 16, "heads": 2, "n_blocks": 2,
                                                                 "hash.orders": [1, 2, 3], "hash.joint": True,
                                                                 "hash.write_mode": modes, "hash.chunk": 16})
    assert scfg["joint"]
    mcfg["vocab"] = 20
    model = Model(mcfg).double()
    tok = _tokens(1, 40, 6, 3)
    t2 = tok.copy()
    t2[0, 25:] = _tokens(1, 15, 6, 4)
    outs = []
    for t in (tok, t2):
        s = sched_to_torch(build_schedule(t, **scfg), "cpu")
        outs.append(model(torch.as_tensor(t), s)[0])
    assert torch.allclose(outs[0][:, :25], outs[1][:, :25], atol=1e-10)
    outs[0].sum().backward()
    kw = model.blocks[0].mixer.kw_proj.weight.grad
    assert kw is not None and torch.isfinite(kw).all() and kw.abs().sum() > 0


@pytest.mark.parametrize("ov", [{"hash.orders": [1, 2]},
                                {"hash.orders": [1, 2], "hash.joint": True, "hash.write_mode": "delta"}])
def test_long_eval_checkpoint_builds_hgdn_schedule(tmp_path, ov):
    """Regression: long_eval_checkpoint dropped scfg for hgdn (no overlay/static), so every long
    eval of the lm-hgdn pilot errored with 'needs the schedule'."""
    from src.longeval import long_eval_checkpoint, needs_schedule

    torch.set_default_dtype(torch.float32)
    mcfg, scfg = arm_config("hgdn", dict(name="lm", L_max=64), {"d_model": 16, "heads": 2, "n_blocks": 1, **ov})
    assert needs_schedule(mcfg)
    from src.text import vocab_size

    mcfg["vocab"] = vocab_size()
    model = Model(mcfg)
    for blk in model.blocks:
        blk.mixer.force_naive = True
    ck = tmp_path / "ck.pt"
    torch.save(dict(model=model.state_dict(), mcfg=mcfg, scfg=scfg), ck)
    spec = dict(stream_lengths=[], probe_lengths=[], passkey_lengths=[128], passkey_n=1)
    out = long_eval_checkpoint(ck, spec)
    assert out["passkey"] and "error" not in out["passkey"][0], out["passkey"]
