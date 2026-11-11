"""Continuous cache (src/cache.py) as an eval-time baseline: correctness against a brute-force
reference, causality, and lam = 0 reproducing the model's own loss exactly."""
import math

import numpy as np
import pytest
import torch

from src.cache import cache_logp_target, cache_logp_vocab, grid_losses, mix_nll


def _brute(hq, qpos, keys, labels, targets, theta, k=None):
    """float64 loop: log p_cache(target) over keys i < t (or their top-k by dot product)."""
    out = []
    for q, t, y in zip(hq.double(), qpos.tolist(), targets.tolist()):
        if t == 0:
            out.append(float("nan"))
            continue
        s = keys[:t].double() @ q
        lab = labels[:t]
        if k is not None and k < t:
            s, idx = s.topk(k)
            lab = lab[idx]
        w = torch.softmax(theta * s, 0)
        p = float(w[lab == y].sum())
        out.append(math.log(p) if p > 0 else float("-inf"))
    return np.array(out)


def _rand(seed, L=37, d=8, nq=15, V=5):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(L, d, generator=g)
    labels = torch.randint(0, V, (L,), generator=g)
    qpos = torch.randperm(L, generator=g)[:nq]
    qpos[0] = 0  # a query with no past
    hq = keys[qpos] + 0.3 * torch.randn(nq, d, generator=g)
    targets = torch.randint(0, V, (nq,), generator=g)
    return hq, qpos, keys, labels, targets


@pytest.mark.parametrize("k", [None, 1, 4, 64])
def test_cache_matches_brute_force(k):
    thetas = [0.1, 0.7, 3.0]
    for seed in range(4):
        hq, qpos, keys, labels, targets = _rand(seed)
        lp, has = cache_logp_target(hq, qpos, keys, labels, targets, thetas, k=k, q_chunk=4, k_chunk=5)
        assert has.tolist() == (qpos > 0).tolist()
        for j, th in enumerate(thetas):
            ref = _brute(hq, qpos, keys, labels, targets, th, k=k)
            m = has.numpy()
            got = lp[j].numpy()[m]
            assert np.array_equal(np.isinf(got), np.isinf(ref[m]))
            fin = np.isfinite(ref[m])
            np.testing.assert_allclose(got[fin], ref[m][fin], atol=1e-5)


@pytest.mark.parametrize("k", [None, 3])
def test_vocab_distribution_matches_target(k):
    hq, qpos, keys, labels, targets = _rand(7, V=6)
    lp, has = cache_logp_target(hq, qpos, keys, labels, targets, [0.5], k=k, k_chunk=6)
    lv = cache_logp_vocab(hq, qpos, keys, labels, 0.5, 6, k=k, k_chunk=6)
    m = has
    assert torch.allclose(lv[m].exp().sum(1), torch.ones(int(m.sum())), atol=1e-5)
    at = lv.gather(1, targets[:, None])[:, 0]
    assert torch.equal(torch.isinf(at[m]), torch.isinf(lp[0][m]))
    fin = torch.isfinite(lp[0]) & m
    assert torch.allclose(at[fin], lp[0][fin], atol=1e-5)


@pytest.mark.parametrize("k", [None, 4])
def test_cache_is_causal(k):
    """Keys and labels at positions >= t never reach the query at t."""
    hq, qpos, keys, labels, targets = _rand(3)
    for j, t in enumerate(qpos.tolist()):
        one = slice(j, j + 1)
        lp, _ = cache_logp_target(hq[one], qpos[one], keys, labels, targets[one], [1.0], k=k, k_chunk=5)
        k2, l2 = keys.clone(), labels.clone()
        k2[t:] = hq[j] * 100  # would dominate any softmax it entered
        l2[t:] = targets[j]
        lp2, _ = cache_logp_target(hq[one], qpos[one], k2, l2, targets[one], [1.0], k=k, k_chunk=5)
        assert torch.equal(lp, lp2)


def test_mix_nll_lam0_exact_and_mixture():
    rng = np.random.default_rng(0)
    lpm = np.log(rng.uniform(0.01, 1, 50))
    lpc = np.log(rng.uniform(0, 1, 50))
    lpc[:5] = -np.inf
    has = np.ones(50, bool)
    has[-3:] = False
    assert np.array_equal(mix_nll(lpm, lpc, has, 0.0), -lpm)
    ref = -np.log(0.7 * np.exp(lpm) + 0.3 * np.exp(lpc))
    ref[~has] = -lpm[~has]
    np.testing.assert_allclose(mix_nll(lpm, lpc, has, 0.3), ref, rtol=1e-12)
    g = grid_losses(lpm, lpc[None], has, [0.5], [0.0, 0.3])
    assert g["0.5"]["0"] == float((-lpm).mean())


def _tiny(arm):
    from src.controls import arm_config
    from src.models import Model

    torch.manual_seed(0)
    if arm == "attn":
        m = Model(dict(mixer="attn", d_model=16, n_blocks=1, heads=2, mlp_hidden=32, max_len=256, vocab=300))
        return m.eval(), {}
    mcfg, scfg = arm_config("ovl_only", {"name": "kv"}, {"overlay.orders": [1, 2]})
    mcfg = dict(mcfg, d_model=16, vocab=300)
    return Model(mcfg).eval(), scfg


@pytest.mark.parametrize("arm", ["attn", "ovl_only"])
def test_tiny_model_cache_vs_brute_force_and_causal(arm):
    from src.longeval import _cached, _nll_at

    model, scfg = _tiny(arm)
    dev = torch.device("cpu")
    rng = np.random.default_rng(1)
    tokens = rng.integers(2, 40, 200).astype(np.int64)  # small alphabet: plenty of label matches
    pos = np.arange(1, 120)
    cache = dict(thetas=[0.05, 0.5], lams=[0.0, 0.2], k=None, theta_idx=1, lam=0.2)
    nll, info = _nll_at(model, scfg, tokens, pos, dev, cache=cache)
    lpc, has = info["cache"]
    with torch.no_grad():
        from src.schedule import build_schedule
        from src.train import sched_to_torch

        st = sched_to_torch(build_schedule(tokens[None], **scfg), dev) if scfg.get("orders") else None
        h = model(torch.as_tensor(tokens[None]), st, return_hidden=True)[0][0].float()
        logp = torch.log_softmax(model.head(h).float(), -1)
    tg = torch.as_tensor(tokens[pos + 1])
    np.testing.assert_allclose(nll, -logp[torch.as_tensor(pos), tg].numpy(), atol=1e-5)
    for j, th in enumerate(cache["thetas"]):
        ref = _brute(h[pos], torch.as_tensor(pos), h[:-1], torch.as_tensor(tokens[1:]), tg, th)
        assert np.array_equal(np.isinf(ref), np.isinf(lpc[j]))
        fin = np.isfinite(ref)
        np.testing.assert_allclose(lpc[j][fin], ref[fin], atol=1e-4)
    # the reported metric is the mixture at (theta_idx, lam)
    mixed, _ = _cached(nll, info, cache)
    pm, pc = np.exp(-nll), np.exp(lpc[1])
    np.testing.assert_allclose(mixed, -np.log(0.8 * pm + 0.2 * pc), rtol=1e-6)
    # causality end to end: change every token after the last target; scored results are unchanged
    t2 = tokens.copy()
    t2[pos[-1] + 2:] = rng.integers(2, 300, len(t2) - pos[-1] - 2)
    nll2, info2 = _nll_at(model, scfg, t2, pos, dev, cache=cache)
    np.testing.assert_allclose(nll2, nll, atol=1e-6)
    np.testing.assert_allclose(info2["cache"][0], lpc, atol=1e-6)


@pytest.mark.parametrize("arm", ["attn", "ovl_only"])
def test_lam0_reproduces_base_loss_and_tuning(arm):
    from src.longeval import doc_eval, passkey_eval, tune_cache

    model, scfg = _tiny(arm)
    dev = torch.device("cpu")
    rng = np.random.default_rng(2)
    books = [rng.integers(2, 60, 230).astype(np.int64) for _ in range(3)]
    base = doc_eval(model, scfg, dev, books, 200)
    zero = dict(thetas=[0.3], lams=[0.0], k=None, theta_idx=0, lam=0.0)
    with_cache = doc_eval(model, scfg, dev, books, 200, cache=zero)
    assert with_cache["loss"] == base["loss"] and with_cache["loss_by_bin"] == base["loss_by_bin"]
    assert with_cache["cache_grid"]["0.3"]["0"] == pytest.approx(base["loss"], abs=1e-6)

    # tuning on separate "validation" books picks a grid point and the test grid includes lam = 0
    val = [rng.integers(2, 60, 230).astype(np.int64) for _ in range(2)]
    tuned = tune_cache(model, scfg, dev, dict(theta_rel=[1.0, 30.0], lams=[0.0, 0.1, 0.5], tune_max_len=200),
                       books=val)
    assert tuned["lam"] in (0.0, 0.1, 0.5) and tuned["theta_rel"] in (1.0, 30.0)
    assert tuned["tune_loss"] <= tuned["tune_loss_nocache"] + 1e-12
    assert tuned["tune_loss"] == min(v for row in tuned["grid"].values() for v in row.values())
    r = doc_eval(model, scfg, dev, books, 200, cache=tuned["eval"])
    t, lam = f"{tuned['theta']:.6g}", f"{tuned['lam']:g}"
    assert r["loss"] == pytest.approx(r["cache_grid"][t][lam], rel=1e-6)
    assert r["cache_grid"][t]["0"] == pytest.approx(base["loss"], abs=1e-6)


def test_passkey_top1_with_cache(monkeypatch):
    """Passkey under the cache: argmax over the full mixture; lam = 0 gives the model's own answers."""
    import src.longeval as le

    model, scfg = _tiny("attn")
    dev = torch.device("cpu")
    monkeypatch.setattr(le, "_encode", lambda s: np.array([2 + (ord(c) % 50) for c in s.replace(" ", "")][:6]))
    filler = np.random.default_rng(0).integers(2, 300, 5000).astype(np.int64)
    base = le.passkey_eval(model, scfg, dev, 128, 3, filler, np.random.default_rng(5))
    zero = dict(thetas=[0.3], lams=[0.0], k=None, theta_idx=0, lam=0.0)
    r0 = le.passkey_eval(model, scfg, dev, 128, 3, filler, np.random.default_rng(5), cache=zero)
    assert r0["acc"] == base["acc"] and r0["answer_nll"] == base["answer_nll"]
    sharp = dict(thetas=[50.0], lams=[0.9], k=None, theta_idx=0, lam=0.9)
    r1 = le.passkey_eval(model, scfg, dev, 128, 3, filler, np.random.default_rng(5), cache=sharp)
    assert 0.0 <= r1["acc"] <= 1.0 and np.isfinite(r1["answer_nll"])


def test_probe_and_stream_with_cache():
    """Copy probes and the stream run under the cache; lam = 0 matches the plain eval exactly and
    a sharp cache lowers the 2nd-copy loss (the copy is in the cache)."""
    from src.longeval import probe_eval, stream_eval

    model, scfg = _tiny("attn")
    dev = torch.device("cpu")
    rng = np.random.default_rng(3)
    filler, held = rng.integers(2, 300, 4000), rng.integers(2, 300, 4000)
    base = probe_eval(model, scfg, dev, 200, 2, 40, filler, held, np.random.default_rng(9))
    grid = dict(thetas=[0.01, 1.0], lams=[0.0, 0.5], k=None)
    zero = probe_eval(model, scfg, dev, 200, 2, 40, filler, held, np.random.default_rng(9),
                      cache=dict(grid, theta_idx=1, lam=0.0))
    assert zero["loss_second"] == base["loss_second"] and zero["gain"] == base["gain"]
    sharp = probe_eval(model, scfg, dev, 200, 2, 40, filler, held, np.random.default_rng(9),
                       cache=dict(grid, theta_idx=1, lam=0.5))
    assert sharp["loss_second"] < base["loss_second"]
    assert sharp["cache_grid_second"]["1"]["0.5"] == pytest.approx(sharp["loss_second"], rel=1e-6)
    s0 = stream_eval(model, scfg, dev, 150, held, 2)
    s1 = stream_eval(model, scfg, dev, 150, held, 2, cache=dict(grid, theta_idx=0, lam=0.0))
    assert s1["loss"] == s0["loss"] and s1["cache_grid"]["0.01"]["0"] == pytest.approx(s0["loss"], abs=1e-6)
