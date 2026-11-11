"""Scan == sequential reference (§9.4), gradcheck (§9.5), causality (§9.6)."""
import math

import numpy as np
import pytest
import torch

from conftest import N_RANDOM_BATCHES
from src.controls import ARMS, arm_config
from src.models import Model, block_mixers, fla_available, fla_status
from src.overlay import Overlay, reference_read
from src.scan import segmented_scan
from src.schedule import build_schedule
from src.tasks import BOS, PAD, KVConfig, kv_batch
from src.train import sched_to_torch


def _random_tokens(rng, B, L, vocab_hi=24):
    toks = rng.integers(2, vocab_hi, size=(B, L))  # small vocab -> many repeats
    toks[:, 0] = BOS
    for b in range(B):
        cut = rng.integers(L // 2, L + 1)
        toks[b, cut:] = PAD
    return toks


@pytest.mark.parametrize("write_mode", ["delta", ("mean", "channel")])
@pytest.mark.parametrize("mode", ["growing", "bounded"])
def test_scan_equals_reference(mode, write_mode):
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    ovl = Overlay(d_model=16, orders=(1, 2), H=2, d_mem=4, write_mode=write_mode)
    for p in ovl.parameters():  # larger weights so differences would show
        torch.nn.init.normal_(p, std=0.5)
    M = 8  # tiny cache with a 6-24 symbol vocab -> forced evictions
    n_evicting = 0
    for i in range(N_RANDOM_BATCHES):
        toks = _random_tokens(rng, 3, int(rng.integers(8, 48)), vocab_hi=int(rng.integers(6, 24)))
        x = torch.randn(*toks.shape, 16)
        s = build_schedule(toks, orders=(1, 2), H=2, mode=mode, M=M, ways=2)
        m_fast, _ = ovl.read(x, sched_to_torch(s, "cpu"))
        m_ref = reference_read(ovl, x, toks, mode=mode, M=M, ways=2)
        assert (m_fast - m_ref).abs().max() < 1e-5, i
        if mode == "bounded":
            g = build_schedule(toks, orders=(1, 2), H=2, mode="growing")
            n_evicting += int(((g["read_ptr"] >= 0) & (s["read_ptr"] < 0)).any())
    if mode == "bounded":
        assert n_evicting > 0  # evictions were actually exercised


def test_scan_matches_loop():
    torch.manual_seed(0)
    seg = [3, 1, 5, 2]
    E = sum(seg)
    a, b = torch.rand(E, 1), torch.randn(E, 2)
    starts = torch.tensor(sum([[sum(seg[:i])] * n for i, n in enumerate(seg)], []))
    out = segmented_scan(a, b, starts, max(seg))
    ref = torch.zeros(E, 2)
    for i in range(E):
        prev = ref[i - 1] if i > starts[i] else torch.zeros(2)
        ref[i] = a[i] * prev + b[i]
    assert torch.allclose(out, ref, atol=1e-6)


def test_gradcheck_scan_path():
    torch.manual_seed(0)
    ovl = Overlay(d_model=6, orders=(1, 2), H=2, d_mem=3).double()
    toks = np.array([[BOS, 5, 6, 5, 6, 5, 7, 5, 6]])
    s = sched_to_torch(build_schedule(toks, orders=(1, 2), H=2), "cpu")
    x = torch.randn(1, 9, 6, dtype=torch.double, requires_grad=True)

    def f(x):
        m, _ = ovl.read(x, s)
        y, _ = ovl.gate(x, m)
        return y

    assert torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-5)
    a = torch.rand(7, 1, dtype=torch.double, requires_grad=True)
    b = torch.randn(7, 2, dtype=torch.double, requires_grad=True)
    st = torch.tensor([0, 0, 0, 3, 4, 4, 4])
    assert torch.autograd.gradcheck(lambda a, b: segmented_scan(a, b, st, 3), (a, b))


CPU_ARMS = [a for a in ARMS if a in ("attn", "ovl_only")]
# On CPU, exercise overlay arms over an attention backbone; with fla, the real arms.
OVERLAY_ON_ATTN = ["ovl", "ovl_bounded", "ovl_multi", "ctrl_random", "ctrl_frozen"]


def _causality(arm, overrides):
    torch.manual_seed(0)
    task = dict(k=2, key_alphabet=16, D=16, C=64, Q=8, L_max=128)
    mcfg, scfg = arm_config(arm, task, overrides)
    if mcfg.get("heads") is None:
        mcfg["heads"] = 2
    if mcfg.get("mlp_hidden") is None:
        mcfg["mlp_hidden"] = 256
    dev = "cuda" if any(m in ("gdn", "gla") for m in block_mixers(mcfg)) else "cpu"
    model = Model(mcfg).to(dev).eval()
    toks = kv_batch(KVConfig(**task), 2, np.random.default_rng(0))["tokens"]
    for t in (20, 60, 100):
        t2 = toks.copy()
        t2[:, t] = 140 + (t2[:, t] + 1) % 100
        outs = []
        for tk in (toks, t2):
            s = sched_to_torch(build_schedule(tk, seed=0, **scfg), dev)
            with torch.no_grad():
                outs.append(model(torch.tensor(tk, device=dev), s)[0])
        assert (outs[0][:, :t] - outs[1][:, :t]).abs().max() < 1e-5, (arm, t)
        assert (outs[0][:, t:] - outs[1][:, t:]).abs().max() > 0  # sanity: change propagates


@pytest.mark.parametrize("arm", CPU_ARMS + OVERLAY_ON_ATTN)
def test_causality_cpu(arm):
    _causality(arm, None if arm in CPU_ARMS else {"mixer": "attn"})


@pytest.mark.parametrize("mixer", [["locker", "attn"], ["attn", "locker"], ["locker", "locker"]])
def test_causality_locker_layers(mixer):
    _causality("ovl", {"mixer": mixer})


def test_locker_layer_wiring():
    task = dict(k=2, key_alphabet=16, D=16, C=64, Q=8, L_max=128)
    m, _ = arm_config("ovl", task, {"mixer": ["attn", "locker"]})
    model = Model(m)
    assert model.overlay_at == [1] and model.blocks[1].mixer is None and model.blocks[0].mixer is not None


@pytest.mark.parametrize("at", [[1], [0, 1, 2]])
def test_causality_placement(at):
    _causality("ovl", {"mixer": "attn", "overlay_at": at})


@pytest.mark.skipif(not fla_available(), reason=str(fla_status()))
@pytest.mark.parametrize("arm", ARMS)
def test_causality_all_arms(arm):
    if arm == "sdm_gdn":  # parked baseline: needs the sparse-delta-memory stack (see docs/OVERNIGHT.md)
        pytest.importorskip("sparse_delta_memory")
    _causality(arm, None)


def test_fla_present_when_required():
    """On GPU hosts the job sets REQUIRE_FLA=1: missing fla must fail, not skip."""
    import os

    if os.environ.get("REQUIRE_FLA"):
        assert fla_available(), fla_status()


def test_force_miss_matches_empty_read():
    """'miss' ablation at a position == the read the model gets when the slot is empty."""
    from src.overlay import _force_miss

    torch.manual_seed(0)
    ovl = Overlay(d_model=8, orders=(2,), H=2, d_mem=3)
    toks = np.array([[BOS, 5, 6, 90, 5, 6]])
    s = sched_to_torch(build_schedule(toks, orders=(2,), H=2), "cpu")
    x = torch.randn(1, 6, 8)
    m, _ = ovl.read(x, s)
    assert m[0, 5].reshape(2, 4)[:, 3].eq(0).all()  # position 5 hits (flag 0)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    mask[0, 5] = True
    mm = _force_miss(m, mask, 3)
    assert torch.equal(mm[0, 5], m[0, 2]) and torch.equal(mm[0, :5], m[0, :5])  # pos 2 is a genuine miss


def test_read_dropout_only_in_training_and_hit_only():
    torch.manual_seed(0)
    toks = np.array([[BOS, 5, 6, 90, 5, 6, 91, 7, 8]])
    s = sched_to_torch(build_schedule(toks, orders=(2,), H=2), "cpu")
    x = torch.randn(1, 9, 8)
    ovl = Overlay(d_model=8, orders=(2,), H=2, d_mem=3, read_dropout=1.0)
    ovl.eval()
    y_eval, _ = ovl(x, s)
    ovl.train()
    y_drop, _ = ovl(x, s)
    ref = Overlay(d_model=8, orders=(2,), H=2, d_mem=3)
    ref.load_state_dict(ovl.state_dict())
    m, _ = ref.read(x, s)
    from src.overlay import _force_miss
    y_all_miss = x + ref.gate(x, _force_miss(m, torch.ones(1, 9, dtype=torch.bool), 3))[0]
    assert torch.allclose(y_drop, y_all_miss, atol=1e-6) and not torch.allclose(y_eval, y_drop)
    h = Overlay(d_model=8, orders=(2,), H=2, d_mem=3, hit_only=True)
    out, _ = h(x, s)
    assert torch.allclose(out[0, :5], x[0, :5]) and not torch.allclose(out[0, 5], x[0, 5])  # pos 5 is the first hit


def test_mean_mode_is_running_mean():
    """Slot read after n writes equals the mean of the n written vectors."""
    torch.manual_seed(0)
    ovl = Overlay(d_model=4, orders=(1,), H=1, d_mem=4, write_mode="mean")
    with torch.no_grad():
        ovl.W_v[0].weight.copy_(torch.eye(4))
    toks = np.array([[BOS, 5, 60, 5, 61, 5, 62, 5]])  # unigram (5) written at 2, 4, 6; read at 7
    s = sched_to_torch(build_schedule(toks, orders=(1,), H=1), "cpu")
    x = torch.randn(1, 8, 4)
    m, _ = ovl.read(x, s)
    assert torch.allclose(m[0, 7, :4], x[0, [2, 4, 6]].mean(0), atol=1e-6) and m[0, 7, 4] == 0


def _notes_reference(ovl, x, toks, K):
    """Loop reference for multi-note reads (growing mode): last K writes to the addressed slot."""
    import math as _m
    from src.hashing import ngram_addresses as _na

    B, L, _ = x.shape
    O, H, dm = len(ovl.orders), ovl.H, ovl.d_mem
    out = torch.zeros(B, L, O, H, dm)
    hitm = torch.zeros(B, L, O, H, dtype=torch.bool)
    for oi, n in enumerate(ovl.orders):
        for h in range(H):
            a, val = _na(toks, n, h)
            for b in range(B):
                for s in range(L):
                    if not val[b, s]:
                        continue
                    ps = [p for p in range(1, s) if val[b, p - 1] and a[b, p - 1] == a[b, s] and toks[b, p] != PAD]
                    if not ps:
                        continue
                    ps = ps[-K:]
                    q = ovl.W_qn[oi](x[b, s])
                    ks = torch.stack([ovl.W_kn[oi](x[b, p]) for p in ps])
                    vs = torch.stack([ovl.W_v[oi](x[b, p]) for p in ps])
                    w = torch.softmax(ks @ q / _m.sqrt(dm), 0)
                    out[b, s, oi, h] = w @ vs
                    hitm[b, s, oi, h] = True
    return out, hitm


@pytest.mark.parametrize("K", [1, 3])
def test_notes_read_matches_loop(K):
    rng = np.random.default_rng(1)
    torch.manual_seed(0)
    ovl = Overlay(d_model=12, orders=(1, 2), H=2, d_mem=4, notes=K)
    for p in ovl.parameters():
        torch.nn.init.normal_(p, std=0.5)
    for _ in range(5):
        toks = _random_tokens(rng, 2, int(rng.integers(10, 30)), vocab_hi=8)
        x = torch.randn(*toks.shape, 12)
        s = sched_to_torch(build_schedule(toks, orders=(1, 2), H=2), "cpu")
        with torch.no_grad():
            m, _ = ovl.read(x, s)
            ref, hitm = _notes_reference(ovl, x, toks, K)
        r = m.reshape(*toks.shape, 2, 2, 5)
        assert torch.allclose(r[..., :4], ref, atol=1e-5)
        assert torch.equal(r[..., 4] == 0, hitm)


def test_age_feature_values():
    ovl = Overlay(d_model=4, orders=(2,), H=1, d_mem=2, age_feature=True)
    toks = np.array([[BOS, 5, 6, 90, 140, 141, 142, 5, 6]])  # write at 3, read at 8 -> age 5
    s = sched_to_torch(build_schedule(toks, orders=(2,), H=1), "cpu")
    m, _ = ovl.read(torch.randn(1, 9, 4), s)
    r = m.reshape(9, 4)
    assert abs(float(r[8, 3].detach()) - math.log1p(5) / 8) < 1e-6 and r[8, 2] == 0
    assert r[2, 2] == 1 and r[2, 3] == 0  # first occurrence: empty, no age


def test_gradcheck_notes_and_age():
    torch.manual_seed(0)
    ovl = Overlay(d_model=6, orders=(1, 2), H=2, d_mem=3, notes=2, age_feature=True).double()
    toks = np.array([[BOS, 5, 6, 5, 6, 5, 7, 5, 6]])
    s = sched_to_torch(build_schedule(toks, orders=(1, 2), H=2), "cpu")
    x = torch.randn(1, 9, 6, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: ovl.gate(x, ovl.read(x, s)[0])[0], (x,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("ov", [{"overlay.notes": 3, "overlay_at": [1]}, {"overlay.age_feature": True},
                                {"overlay.notes": 2, "overlay.age_feature": True, "overlay.read_dropout": 0.5}])
def test_causality_new_options(ov):
    _causality("ovl", {"mixer": "attn", **ov})


def test_skipgram_heads_recover_corrupted_key():
    """k=4 key; query with its last token corrupted. Contiguous order 4 misses; the skip-gram
    head that drops the final position hits the right write."""
    key, bad = [10, 11, 12, 13], [10, 11, 12, 40]
    toks = np.array([[BOS, *key, 90, 140, 141, *bad]])
    s = build_schedule(toks, orders=(4, (1, 2, 3)), H=1)
    L = toks.shape[1]
    assert s["read_ptr"][0, L - 1, 0, 0] < 0  # exact 4-gram: miss
    p = s["read_ptr"][0, L - 1, 1, 0]
    assert p >= 0 and s["ev_pos"][p] == 5  # skip-gram: hits the write at the value position


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flash_attention_kernel_is_used_at_4k():
    """Forcing the FlashAttention backend must work at our shapes (bf16 autocast, 4096 tokens);
    if it can't, sdpa_kernel raises here instead of the run silently using the math path."""
    from src.models import CausalAttention

    att = CausalAttention(256, 4, conv=4).cuda()
    x = torch.randn(2, 4096, 256, device="cuda", requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = att(x)
    y.float().sum().backward()
    assert torch.cuda.max_memory_allocated() < 2 * 2**30  # the math path would need several GB here


def test_ngh_is_uniform_average_without_gate():
    """n-gram-head baseline: y = W_O(mean of matched notes); no gate; zero when nothing matched."""
    torch.manual_seed(0)
    ovl = Overlay(d_model=4, orders=(1,), H=1, d_mem=4, write_mode="mean", gate=False, empty_flag=False)
    with torch.no_grad():
        ovl.W_v[0].weight.copy_(torch.eye(4))
    toks = np.array([[BOS, 5, 60, 5, 61, 5, 62, 5, 7]])
    s = sched_to_torch(build_schedule(toks, orders=(1,), H=1), "cpu")
    x = torch.randn(1, 9, 4)
    out, st = ovl(x, s)
    expect = ovl.gate.W_O(torch.cat([x[0, [2, 4, 6]].mean(0), torch.zeros(1)]))
    assert torch.allclose(out[0, 7] - x[0, 7], expect, atol=1e-6)
    assert torch.allclose(out[0, 8], x[0, 8])  # token 7 never seen: nothing added
    assert torch.equal(st["alpha"], torch.ones_like(st["alpha"]))


def test_rope_scores_depend_only_on_offset():
    from src.models import apply_rope

    q = torch.randn(1, 1, 1, 8).expand(1, 1, 6, 8).contiguous()
    k = torch.randn(1, 1, 1, 8).expand(1, 1, 6, 8).contiguous()
    s = apply_rope(q) @ apply_rope(k).transpose(-1, -2)
    assert torch.allclose(s[0, 0, 3, 1], s[0, 0, 4, 2], atol=1e-5) and torch.allclose(s[0, 0, 5, 5], s[0, 0, 0, 0], atol=1e-5)


@pytest.mark.parametrize("ov", [{"mixer": "attn", "rope": True, "pos_emb": False},
                                {"mixer": ["attn", "locker"], "rope": True, "pos_emb": False}])
def test_causality_rope_and_hybrid(ov):
    _causality("ovl", ov)


def test_causality_ngh():
    _causality("ngh123", {"mixer": "attn"})


@pytest.mark.parametrize("modes", ["delta", "channel", "mean", ("delta", "mean", "channel")])
def test_gather_path_matches_event_values(modes):
    """read() projects per position then gathers (memory-light at 1M tokens); values and
    gradients must equal the per-event formula exactly (float64)."""
    from src.schedule import build_schedule
    from src.train import sched_to_torch

    torch.manual_seed(0)
    toks = np.random.default_rng(0).integers(2, 40, size=(2, 200))
    orders = (1, 2, 3) if isinstance(modes, tuple) else (2, 3)
    ov = Overlay(64, orders=orders, H=2, d_mem=16, write_mode=list(modes) if isinstance(modes, tuple) else modes).double()
    s = sched_to_torch(build_schedule(toks, orders=orders, H=2, mode="growing"), torch.device("cpu"))
    x = torch.randn(2, 200, 64, requires_grad=True, dtype=torch.float64)
    xf = x.reshape(-1, 64)
    cnt = torch.arange(s["ev_seg_start"].shape[0]) - s["ev_seg_start"] + 1
    outs = []
    for f in (lambda: ov.event_values(xf[s["ev_flat"]], s["ev_order"], cnt),
              lambda: ov._event_values_gather(xf, s["ev_flat"], s["ev_order"], cnt)):
        v, b = f()
        outs.append((v, b, torch.autograd.grad((v * b).sum() + b.pow(2).sum(), [x, *ov.parameters()], allow_unused=True)))
    (v1, b1, g1), (v2, b2, g2) = outs
    assert torch.allclose(v1, v2, atol=1e-12) and torch.allclose(b1, b2, atol=1e-12)
    for a, b in zip(g1, g2):
        assert (a is None and b is None) or torch.allclose(a, b, rtol=1e-10, atol=1e-10)


def test_duplicate_orders_mixed_write_matches_reference():
    """orders (2, 2) with write_mode [mean, delta]: two separate tables for the same cue, one
    averaging and one delta (the mixed-write locker). Must equal the sequential reference."""
    rng = np.random.default_rng(1)
    torch.manual_seed(0)
    ovl = Overlay(d_model=16, orders=(1, 2, 1, 2), H=1, d_mem=4, write_mode=["mean", "mean", "delta", "delta"])
    for p in ovl.parameters():
        torch.nn.init.normal_(p, std=0.5)
    for _ in range(20):
        toks = _random_tokens(rng, 3, int(rng.integers(8, 48)), vocab_hi=int(rng.integers(6, 24)))
        x = torch.randn(*toks.shape, 16)
        s = build_schedule(toks, orders=(1, 2, 1, 2), H=1, mode="growing")
        m_fast, _ = ovl.read(x, sched_to_torch(s, "cpu"))
        m_ref = reference_read(ovl, x, toks, mode="growing")
        assert torch.allclose(m_fast, m_ref, atol=1e-5)
        mean_part, delta_part = m_fast[..., :2 * 5], m_fast[..., 2 * 5:]
        assert not torch.allclose(mean_part, delta_part)  # the two tables really differ


@pytest.mark.parametrize("kappa_mode", ["row", "stream"])
@pytest.mark.parametrize("log_kappa", [-30.0, 0.0, 30.0])
def test_bayesian_locker_posterior(log_kappa, kappa_mode):
    """prior_rows: read = w * prior[row] + (1 - w) * plain_read on hits, prior[row] on misses,
    w = kappa / (kappa + n), n counted by brute force (writes to the same cue strictly before the
    read). kappa -> 0 recovers plain locker on hits; kappa -> inf reads the prior everywhere."""
    from src.schedule import addresses

    rng = np.random.default_rng(3)
    torch.manual_seed(0)
    orders, H, dm, R = (1, 2), 2, 4, 64
    bay = Overlay(d_model=16, orders=orders, H=H, d_mem=dm, write_mode="mean", prior_rows=R, kappa_mode=kappa_mode)
    for p in bay.parameters():
        torch.nn.init.normal_(p, std=0.5)
    with torch.no_grad():
        (bay.kappa_table if kappa_mode == "row" else bay.kappa_b).fill_(log_kappa)
    plain = Overlay(d_model=16, orders=orders, H=H, d_mem=dm, write_mode="mean")
    plain.load_state_dict({k: v for k, v in bay.state_dict().items() if not (k.endswith("table") or k.startswith("kappa"))})
    for _ in range(10):
        toks = _random_tokens(rng, 2, int(rng.integers(8, 40)), vocab_hi=int(rng.integers(6, 14)))
        x = torch.randn(*toks.shape, 16)
        s = build_schedule(toks, orders=orders, H=H, mode="growing", table_rows=R)
        st = sched_to_torch(s, "cpu")
        got = bay.read(x, st)[0].reshape(*toks.shape, len(orders) * H, dm + 1)
        base = plain.read(x, st)[0].reshape(*toks.shape, len(orders) * H, dm + 1)
        addr, valid, w_addr, w_valid = addresses(toks, orders, H)
        B, L = toks.shape
        for b in range(B):
            for t in range(L):
                for o in range(len(orders)):
                    for h in range(H):
                        k = o * H + h
                        if not valid[b, t, o, h]:
                            assert torch.allclose(got[b, t, k, :dm], torch.zeros(dm), atol=1e-6)
                            continue
                        n = int(sum(w_valid[b, p, o, h] and w_addr[b, p, o, h] == addr[b, t, o, h] for p in range(t)))
                        kap = math.exp(log_kappa)
                        w = kap / (kap + n)
                        prior = bay.prior_table[k, s["read_row"][b, t, o, h]]
                        want = w * prior + (1 - w) * base[b, t, k, :dm]
                        assert torch.allclose(got[b, t, k, :dm], want, atol=1e-5), (b, t, o, h, n)
                        assert got[b, t, k, dm] == base[b, t, k, dm]  # empty flag still marks local misses


def test_bayesian_locker_prior_miss_only():
    """prior_miss_only: hits read the plain local mean, cue misses read the prior."""
    rng = np.random.default_rng(4)
    torch.manual_seed(0)
    bay = Overlay(d_model=16, orders=(1, 2), H=1, d_mem=4, write_mode="mean", prior_rows=64, prior_miss_only=True)
    for p in bay.parameters():
        torch.nn.init.normal_(p, std=0.5)
    plain = Overlay(d_model=16, orders=(1, 2), H=1, d_mem=4, write_mode="mean")
    plain.load_state_dict({k: v for k, v in bay.state_dict().items() if not k.endswith("table")})
    toks = _random_tokens(rng, 2, 30, vocab_hi=10)
    x = torch.randn(*toks.shape, 16)
    s = build_schedule(toks, orders=(1, 2), H=1, mode="growing", table_rows=64)
    st = sched_to_torch(s, "cpu")
    got = bay.read(x, st)[0].reshape(2, 30, 2, 5)
    base = plain.read(x, st)[0].reshape(2, 30, 2, 5)
    hit = base[..., 4] < 0.5
    valid = torch.as_tensor(s["read_row"] >= 0).reshape(2, 30, 2)
    assert torch.allclose(got[hit][:, :4], base[hit][:, :4], atol=1e-6)
    miss = valid & ~hit
    k = torch.arange(2).expand(2, 30, 2)
    want = bay.prior_table[k[miss], torch.as_tensor(s["read_row"]).reshape(2, 30, 2)[miss]]
    assert miss.any() and torch.allclose(got[miss][:, :4], want, atol=1e-6)


def test_bayesian_kappa_prior_mode():
    """kappa_mode "prior": log kappa = b[stream] + w[stream] . prior_row; dense gradients reach w, b."""
    torch.manual_seed(0)
    ov = Overlay(d_model=16, orders=(1, 2), H=1, d_mem=4, write_mode="mean", prior_rows=32, kappa_mode="prior")
    for p in ov.parameters():
        torch.nn.init.normal_(p, std=0.5)
    st, rw = torch.tensor([0, 1, 1]), torch.tensor([3, 5, 7])
    pr = ov.prior_table[st, rw]
    want = ov.kappa_b[st] + (pr * ov.kappa_w[st]).sum(-1)
    assert torch.allclose(ov.log_kappa(st, rw, pr), want)
    toks = _random_tokens(np.random.default_rng(0), 2, 24, vocab_hi=8)
    s = build_schedule(toks, orders=(1, 2), H=1, mode="growing", table_rows=32)
    out = ov.read(torch.randn(2, 24, 16), sched_to_torch(s, "cpu"))[0]
    out.sum().backward()
    assert ov.kappa_w.grad.abs().sum() > 0 and ov.kappa_b.grad.abs().sum() > 0


def _brute_counts(toks, orders, H):
    """n[b, t, o, h] = writes to the cue read at (b, t, o, h) strictly before t (read before write)."""
    from src.schedule import addresses

    addr, valid, w_addr, w_valid = addresses(toks, orders, H)
    B, L = toks.shape
    n = np.zeros((B, L, len(orders), H), dtype=np.int64)
    for b in range(B):
        for t in range(L):
            for o in range(len(orders)):
                for h in range(H):
                    if valid[b, t, o, h]:
                        n[b, t, o, h] = sum(w_valid[b, p, o, h] and w_addr[b, p, o, h] == addr[b, t, o, h] for p in range(t))
    return n


def test_count_feature_matches_brute_force():
    rng = np.random.default_rng(5)
    torch.manual_seed(0)
    orders, H, dm = (1, 2), 2, 4
    ov = Overlay(d_model=16, orders=orders, H=H, d_mem=dm, write_mode="mean", count_feature=True)
    for _ in range(5):
        toks = _random_tokens(rng, 2, int(rng.integers(8, 40)), vocab_hi=int(rng.integers(5, 12)))
        s = build_schedule(toks, orders=orders, H=H, mode="growing")
        r = ov.read(torch.randn(*toks.shape, 16), sched_to_torch(s, "cpu"))[0].reshape(*toks.shape, len(orders) * H, dm + 2)
        n = torch.as_tensor(_brute_counts(toks, orders, H)).reshape(*toks.shape, -1).float()
        assert torch.allclose(r[..., dm + 1], torch.log1p(n) / 8.0, atol=1e-6)
        assert torch.equal(r[..., dm] > 0.5, n == 0)  # empty flag exactly where no earlier note


def test_backoff_keeps_longest_hit():
    rng = np.random.default_rng(6)
    torch.manual_seed(0)
    orders, H, dm = (1, 2, 3, 1, 2, 3), 1, 4  # duplicate lengths (mixed-write layout)
    kw = dict(d_model=16, orders=orders, H=H, d_mem=dm, write_mode=["mean"] * 3 + ["delta"] * 3)
    bo = Overlay(**kw, backoff=True)
    plain = Overlay(**kw)
    plain.load_state_dict(bo.state_dict())
    lens = np.array([1, 2, 3, 1, 2, 3])
    for _ in range(5):
        toks = _random_tokens(rng, 2, int(rng.integers(10, 40)), vocab_hi=int(rng.integers(4, 9)))
        s = build_schedule(toks, orders=orders, H=H, mode="growing")
        st = sched_to_torch(s, "cpu")
        x = torch.randn(*toks.shape, 16)
        got = bo.read(x, st)[0].reshape(*toks.shape, 6, dm + 1)
        base = plain.read(x, st)[0].reshape(*toks.shape, 6, dm + 1)
        hit = (base[..., dm] < 0.5).numpy()
        longest = np.where(hit, lens, 0).max(-1, keepdims=True)
        keep = hit & (lens == longest)
        k = torch.as_tensor(keep)
        assert torch.allclose(got[k][:, :dm], base[k][:, :dm], atol=1e-6)
        assert torch.all(got[~k][:, :dm] == 0) and torch.all(got[~k][:, dm] == 1)


@pytest.mark.skipif(not fla_available(), reason=str(fla_status()))
def test_gdn_segmented_inference_matches_full():
    """Segmented GDN inference (conv + recurrent state carried through fla's Cache) equals one
    full-length pass; used for multi-million-token evals."""
    from src.models import FlaMixer

    torch.manual_seed(0)
    mix = FlaMixer("gdn", 128, 2).cuda().eval()
    x = torch.randn(1, 4096, 128, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = mix(x).float()
        mix.segment = 1000  # uneven segments on purpose
        seg = mix(x).float()
    assert torch.allclose(full, seg, atol=3e-2, rtol=3e-2), (full - seg).abs().max()
    assert (full - seg).abs().mean() < 3e-3


def test_soft_backoff_flags_longest_and_keeps_reads():
    rng = np.random.default_rng(7)
    torch.manual_seed(0)
    orders, H, dm = (1, 2, 3, 1, 2, 3), 1, 4
    kw = dict(d_model=16, orders=orders, H=H, d_mem=dm, write_mode=["mean"] * 3 + ["delta"] * 3)
    soft = Overlay(**kw, backoff="soft")
    plain = Overlay(**kw)
    # read() doesn't use the gate (whose input is one channel wider with the soft flag)
    plain.load_state_dict({k: v for k, v in soft.state_dict().items() if not k.startswith("gate.")}, strict=False)
    lens = np.array([1, 2, 3, 1, 2, 3])
    toks = _random_tokens(rng, 2, 30, vocab_hi=6)
    s = build_schedule(toks, orders=orders, H=H, mode="growing")
    st = sched_to_torch(s, "cpu")
    x = torch.randn(*toks.shape, 16)
    got = soft.read(x, st)[0].reshape(*toks.shape, 6, dm + 2)
    base = plain.read(x, st)[0].reshape(*toks.shape, 6, dm + 1)
    assert torch.allclose(got[..., : dm + 1], base, atol=1e-6)  # every read kept
    hit = (base[..., dm] < 0.5).detach().numpy()
    longest = np.where(hit, lens, 0).max(-1, keepdims=True)
    assert np.array_equal(got[..., dm + 1].detach().numpy() > 0.5, hit & (lens == longest))


def test_selective_backoff_only_masks_flagged_streams():
    """backoff=[...]: hard backoff among flagged streams (delta tables); others always read."""
    rng = np.random.default_rng(8)
    torch.manual_seed(0)
    orders, H, dm = (1, 2, 3, 1, 2, 3), 1, 4
    sel = [False, False, False, True, True, True]
    kw = dict(d_model=16, orders=orders, H=H, d_mem=dm, write_mode=["mean"] * 3 + ["delta"] * 3)
    bo = Overlay(**kw, backoff=sel)
    plain = Overlay(**kw)
    plain.load_state_dict(bo.state_dict())
    lens, selv = np.array([1, 2, 3, 1, 2, 3]), np.array(sel)
    for _ in range(5):
        toks = _random_tokens(rng, 2, int(rng.integers(10, 40)), vocab_hi=int(rng.integers(4, 9)))
        s = build_schedule(toks, orders=orders, H=H, mode="growing")
        st = sched_to_torch(s, "cpu")
        x = torch.randn(*toks.shape, 16)
        got = bo.read(x, st)[0].reshape(*toks.shape, 6, dm + 1).detach()
        base = plain.read(x, st)[0].reshape(*toks.shape, 6, dm + 1).detach()
        hit = (base[..., dm] < 0.5).numpy()
        longest = np.where(hit & selv, lens, 0).max(-1, keepdims=True)
        keep = torch.as_tensor(hit & (~selv | (lens == longest)))
        assert torch.allclose(got[keep][:, :dm], base[keep][:, :dm], atol=1e-6)
        assert torch.all(got[~keep][:, :dm] == 0) and torch.all(got[~keep][:, dm] == 1)


def test_set_backoff_at_inference_matches_trained_with_backoff():
    """Switching backoff on a model built without it reads exactly like one built with it."""
    torch.manual_seed(0)
    kw = dict(d_model=16, orders=(1, 2, 3, 1, 2, 3), H=1, d_mem=4, write_mode=["mean"] * 3 + ["delta"] * 3)
    toks = _random_tokens(np.random.default_rng(9), 2, 30, vocab_hi=6)
    st = sched_to_torch(build_schedule(toks, orders=kw["orders"], H=1, mode="growing"), "cpu")
    x = torch.randn(2, 30, 16)
    for b in (True, [False] * 3 + [True] * 3):
        ref = Overlay(**kw, backoff=b)
        sw = Overlay(**kw)
        sw.load_state_dict(ref.state_dict())
        sw.set_backoff(b)
        assert torch.equal(sw.read(x, st)[0], ref.read(x, st)[0])
        plain = Overlay(**kw)
        plain.load_state_dict(ref.state_dict())
        sw.set_backoff(False)  # and back off again
        assert sw.backoff is False and torch.equal(sw.read(x, st)[0], plain.read(x, st)[0])


def test_overlay_shared_table_ties_one_parameter():
    from src.models import Model

    cfg = dict(mixer=["attn"] * 4, d_model=16, n_blocks=4, heads=2, mlp_hidden=32, max_len=32,
               overlay=dict(kind="memory", orders=(1, 2), H=1, d_mem=8, prior_rows=64, prior_add=True),
               overlay_at=[1, 2, 3])
    sep = Model(cfg)
    sh = Model(dict(cfg, overlay_shared_table=True))
    tables = [o.prior_table for o in sh.overlays.values()]
    assert all(t is tables[0] for t in tables)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(sep) - n(sh) == 2 * tables[0].numel()
    assert sum(1 for name, _ in sh.named_parameters() if name.endswith("prior_table")) == 1
