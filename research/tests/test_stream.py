"""Streaming decode (src/stream.py) == the parallel forward, token by token, in float64."""
import numpy as np
import pytest
import torch

from src.controls import arm_config
from src.hashing import ngram_addresses
from src.models import Model
from src.schedule import build_schedule, mix64
from src.stream import CueHasher, mix64_int
from src.tasks import BOS, PAD
from src.train import sched_to_torch

TOL = 1e-9


def _tokens(rng, B, L, vocab_hi=12, pad=True):
    toks = rng.integers(2, vocab_hi, size=(B, L))  # tiny vocab: many repeated N-grams
    toks[:, 0] = BOS
    if pad:  # a few PAD tokens mid-stream exercise the "no write on PAD" rule
        toks[rng.random((B, L)) < 0.05] = PAD
        toks[:, 0] = BOS
    return toks


def _model(arm, overrides, seed=0, L=64):
    mcfg, scfg = arm_config(arm, {"k": 2, "L_max": L}, overrides)
    torch.manual_seed(seed)
    model = Model(mcfg).double().eval()
    for name, p in model.named_parameters():  # larger weights so mismatches would show
        if "norm" not in name:
            torch.nn.init.normal_(p, std=0.3)
    return model, scfg


def _parallel(model, scfg, toks):
    sched = None
    if scfg and (model.overlays or model.static is not None):
        sched = sched_to_torch(build_schedule(toks, **scfg), "cpu")
    with torch.no_grad():
        return model(torch.as_tensor(toks), sched)[0]


# (arm, overrides): every write mode, the gate, H, the static table, read features, placements.
CASES = {
    "delta": ("ovl_only", {}),
    "channel": ("ovl_only", {"overlay.write_mode": "channel"}),
    "mean": ("ovl_only", {"overlay.write_mode": "mean"}),
    "mixed_o1-3_static": ("ovl_only", {"overlay.orders": [1, 2, 3, 1, 2, 3], "overlay.H": 1,
                                       "overlay.write_mode": ["mean"] * 3 + ["delta"] * 3,
                                       "static_rows": 64}),
    "ngh_nogate": ("ngh123", {"mixer": "none"}),
    "fixed_beta_H3": ("ovl_only", {"overlay.fixed_beta": 0.3, "overlay.H": 3}),
    "skipgram": ("ovl_only", {"overlay.orders": [2, [0, 2]], "overlay.write_mode": ["delta", "mean"]}),
    "features": ("ovl_only", {"overlay.orders": [1, 2], "overlay.age_feature": True,
                              "overlay.count_feature": True, "overlay.hit_only": True}),
    "backoff_hard": ("ovl_only", {"overlay.orders": [1, 2, 3], "overlay.backoff": True}),
    "backoff_soft": ("ovl_only", {"overlay.orders": [1, 2, 3], "overlay.backoff": "soft"}),
    "backoff_select": ("ovl_only", {"overlay.orders": [1, 2, 1, 2], "overlay.H": 1,
                                    "overlay.write_mode": ["mean", "mean", "delta", "delta"],
                                    "overlay.backoff": [False, False, True, True]}),
    "prior_row": ("ovl_only", {"overlay.orders": [1, 2], "overlay.prior_rows": 32,
                               "overlay.write_mode": "mean"}),
    "prior_kappa_prior": ("ovl_only", {"overlay.prior_rows": 32, "overlay.kappa_mode": "prior"}),
    "prior_miss_only": ("ovl_only", {"overlay.prior_rows": 32, "overlay.prior_miss_only": True}),
    "notes": ("ovl_only", {"overlay.orders": [1, 2], "overlay.notes": 3}),
    "bounded": ("ovl_only", {"overlay.orders": [1, 2], "sched.mode": "bounded", "sched.M": 8,
                             "sched.ways": 2, "overlay.write_mode": ["mean", "delta"]}),
    "attn_learned_pos": ("attn", {"overlay": {"kind": "memory", "orders": (2,), "H": 2, "d_mem": 8}}),
    "attn_rope_conv_mixed": ("attn", {"rope": True, "pos_emb": False, "attn_conv": 4,
                                      "overlay": {"kind": "memory", "orders": (1, 2, 1, 2), "H": 1, "d_mem": 8,
                                                  "write_mode": ["mean", "mean", "delta", "delta"]},
                                      "static_rows": 64}),
    "attn_overlay_after_blocks": ("attn", {"rope": True, "pos_emb": False, "overlay_at": [0, 2],
                                           "overlay": {"kind": "memory", "orders": (1, 2), "H": 2, "d_mem": 8}}),
    "locker_block_attn": ("attn", {"mixer": ["locker", "attn"], "rope": True, "pos_emb": False,
                                   "overlay": {"kind": "memory", "orders": (2,), "H": 2, "d_mem": 8,
                                               "write_mode": "mean"}}),
    "attn_no_locker": ("attn", {"rope": True, "pos_emb": False, "attn_conv": 3}),
}


def _build(case, seed=0):
    arm, ov = CASES[case]
    ov = dict(ov)
    if arm in ("ovl_only", "ngh123"):
        ov = {"d_model": 16, "mlp_hidden": 24, "overlay.d_mem": 8, **ov}
    else:
        ov = {"d_model": 16, "heads": 2, "mlp_hidden": 24, **ov}
    return _model(arm, ov, seed)


@pytest.mark.parametrize("case", sorted(CASES))
def test_stream_matches_parallel(case):
    model, scfg = _build(case)
    rng = np.random.default_rng(1)
    toks = _tokens(rng, 2, 48)
    ref = _parallel(model, scfg, toks)
    dec = model.decoder(batch=2, scfg=scfg)
    out = torch.stack([dec.step(torch.as_tensor(toks[:, t])) for t in range(toks.shape[1])], 1)
    assert (out - ref).abs().max() < TOL * max(1.0, ref.abs().max())


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("P", [1, 5, 30])
def test_prefill_then_stream(case, P):
    model, scfg = _build(case, seed=2)
    rng = np.random.default_rng(P)
    toks = _tokens(rng, 2, 44)
    ref = _parallel(model, scfg, toks)
    dec = model.decoder(batch=2, scfg=scfg)
    pre = dec.prefill(torch.as_tensor(toks[:, :P]), return_all=True)
    assert (pre - ref[:, :P]).abs().max() < TOL * max(1.0, ref.abs().max())
    out = torch.stack([dec.step(toks[:, t].tolist()) for t in range(P, toks.shape[1])], 1)
    assert (out - ref[:, P:]).abs().max() < TOL * max(1.0, ref.abs().max())


def test_stream_float32_close():
    """The production dtype: float32 stream vs float32 parallel forward."""
    model, scfg = _build("mixed_o1-3_static")
    model = model.float()
    rng = np.random.default_rng(3)
    toks = _tokens(rng, 1, 200, vocab_hi=8, pad=False)
    ref = _parallel(model, scfg, toks)
    dec = model.decoder(batch=1, scfg=scfg)
    dec.prefill(torch.as_tensor(toks[:, :100]))
    out = torch.stack([dec.step(toks[:, t]) for t in range(100, 200)], 1)
    assert (out - ref[:, 100:]).abs().max() < 1e-4 * ref.abs().max()


def test_state_bytes_counts_slots():
    model, scfg = _build("attn_rope_conv_mixed")
    rng = np.random.default_rng(4)
    toks = _tokens(rng, 1, 40, pad=False)
    dec = model.decoder(batch=1, scfg=scfg)
    dec.prefill(torch.as_tensor(toks[:, :30]))
    for t in range(30, 40):
        dec.step(toks[:, t])
    sb = dec.state_bytes()
    sched = build_schedule(toks, **scfg)
    assert sb["locker_slots"] == int(sched["n_slots"].sum())
    assert sb["kv"] == 2 * 2 * 40 * 16 * 8  # 2 blocks x (k, v) x 40 tokens x d=16 x float64
    # the synthetic (timing-only) state has the same slot set
    dec2 = model.decoder(batch=1, scfg=scfg)
    dec2.prefill_synthetic(torch.as_tensor(toks))
    assert dec2.state_bytes()["locker_slots"] == sb["locker_slots"]
    assert dec2.step([5]).shape == (1, model.head.out_features)


def test_hasher_matches_numpy():
    rng = np.random.default_rng(5)
    toks = _tokens(rng, 2, 60)
    orders = (1, 3, (0, 2, 3))
    hs = CueHasher(orders, 2, 2)
    got = [hs.push(toks[:, t])[0] for t in range(60)]
    j = 0
    for n in orders:
        for h in range(2):
            a, v = ngram_addresses(toks, n, h)
            for b in range(2):
                for t in range(60):
                    g = got[t][b][j]
                    assert (g is not None) == bool(v[b, t])
                    if g is not None:
                        assert g == int(a[b, t])
                        assert mix64_int(g) == int(mix64(np.int64(a[b, t])))
            j += 1


BENCH_SMALL = dict(n_decode=4, warmup=1, d_model=16, n_blocks=2, heads=2, mlp_hidden=32, vocab=64,
                   tokens="random")
ATTN_LOCKER = dict(arm="ovl", overrides={"mixer": "attn", "rope": True, "pos_emb": False, "static_rows": 1 << 10,
                                         "overlay.orders": [1, 2, 1, 2], "overlay.H": 1,
                                         "overlay.write_mode": ["mean", "mean", "delta", "delta"]})


def test_bench_decode_cpu_smoke():
    """The benchmark runs end to end on CPU (attention arms only: GDN needs fla + CUDA)."""
    from src.bench_decode import bench

    res = bench(dict(BENCH_SMALL, lengths=[64, 128], arms={
        "attn": dict(arm="attn", overrides={"rope": True, "pos_emb": False, "attn_conv": 4}),
        "attn_locker": ATTN_LOCKER, "gdn": dict(arm="base", overrides={})}))
    assert res["ok"]
    r = res["results"]
    assert r["attn"]["64"]["decode_tok_s"] > 0
    assert r["attn_locker"]["128"]["state"]["locker_slots"] > 0
    assert "error" in r["gdn"]["64"] or r["gdn"]["64"].get("decode_tok_s")
    # the OOM fallback (timing-only synthetic state) holds the same state, up to the slots written by
    # the greedy decode (its tokens differ); exact equality is in test_state_bytes_counts_slots
    syn = bench(dict(BENCH_SMALL, lengths=[128], prefill="synthetic", arms={"attn_locker": ATTN_LOCKER}))
    s, p = syn["results"]["attn_locker"]["128"]["state"], r["attn_locker"]["128"]["state"]
    assert syn["results"]["attn_locker"]["128"]["prefill"] == "synthetic"
    assert s["kv"] == p["kv"]
    assert abs(s["locker_slots"] - p["locker_slots"]) <= 4 * (BENCH_SMALL["n_decode"] + BENCH_SMALL["warmup"])


@pytest.mark.parametrize("mixer", ["gdn", ["gdn", "attn"], "gla"])
def test_stream_fla_matches_parallel(mixer):
    """GPU only: fla's recurrent step (Cache) after a chunked prefill == the parallel forward.
    float32 on CUDA; chunk and recurrent kernels sum in different orders, so a looser tolerance."""
    from src.models import fla_available, fla_status

    if not fla_available():
        pytest.skip(fla_status())
    mcfg, scfg = arm_config("ovl", {"k": 2, "L_max": 256},
                            {"mixer": mixer, "d_model": 128, "n_blocks": 2, "heads": 2, "mlp_hidden": 256,
                             "rope": True, "pos_emb": False, "attn_conv": 4, "static_rows": 1 << 10,
                             "overlay.orders": [1, 2, 3, 1, 2, 3], "overlay.H": 1,
                             "overlay.write_mode": ["mean"] * 3 + ["delta"] * 3})
    torch.manual_seed(0)
    model = Model(mcfg).cuda().eval()
    from src.models import CausalAttention

    for m in model.modules():  # fp32 attention on both sides (forced flash would cast to bf16)
        if isinstance(m, CausalAttention):
            m.force_flash = False
    toks = _tokens(np.random.default_rng(0), 2, 200, vocab_hi=40, pad=False)
    with torch.no_grad():
        sched = sched_to_torch(build_schedule(toks, **scfg), "cuda")
        ref = model(torch.as_tensor(toks, device="cuda"), sched)[0]
    dec = model.decoder(batch=2, scfg=scfg)
    pre = dec.prefill(torch.as_tensor(toks[:, :120], device="cuda"), return_all=True)
    out = torch.stack([dec.step(torch.as_tensor(toks[:, t], device="cuda")) for t in range(120, 200)], 1)
    scale = ref.abs().max()
    assert (pre - ref[:, :120]).abs().max() < 2e-3 * scale
    assert (out - ref[:, 120:]).abs().max() < 2e-3 * scale
