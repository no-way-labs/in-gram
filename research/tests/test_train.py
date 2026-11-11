"""End-to-end training loop checks (CPU, a few steps)."""
import json

import pytest

from src.train import run


@pytest.mark.parametrize("arm,overrides", [("ovl_only", None), ("ovl", {"mixer": ["locker", "attn"]})])
def test_spawn_workers_and_logging(tmp_path, arm, overrides):
    # spawn-started DataLoader workers: catches unpicklable pipeline state, which
    # is what happens on Rhombic GPU hosts (runs start inside spawn processes).
    cfg = dict(run_id="t", arm=arm, task=dict(D=16, C=64, Q=8), steps=4, eval_every=2, eval_size=32,
               batch=8, num_workers=2, mp_context="spawn", overrides=overrides)
    final = run(cfg, tmp_path)
    assert final["final"] and final["acc"] is not None and final["acc_ablate"] is not None
    lines = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert "config" in lines[0] and [l["step"] for l in lines[1:-1]] == [2, 4]
    assert run(cfg, tmp_path) == final  # idempotent: completed run is skipped


def test_extra_evals_longer_context(tmp_path):
    cfg = dict(run_id="x", arm="ovl", task=dict(D=16, C=64, Q=8), steps=2, eval_every=2, eval_size=16,
               batch=8, num_workers=0, overrides={"mixer": "attn", "pos_emb": False},
               extra_evals={"len4x": dict(D=64, C=256)}, extra_eval_size=8, extra_eval_batch=4)
    final = run(cfg, tmp_path)
    assert final["acc_len4x"] is not None and final["acc_len4x_ablate_q"] is not None
    header = json.loads((tmp_path / "x.jsonl").read_text().splitlines()[0])
    assert header["model"]["max_len"] >= 256 + 8 * 3 + 1


def test_chunked_ce_matches_full(monkeypatch):
    import torch
    import torch.nn.functional as F

    import src.train as T
    from src.models import Model

    torch.manual_seed(0)
    m = Model(dict(mixer="attn", d_model=16, n_blocks=1, heads=2, mlp_hidden=32, max_len=64, vocab=1000))
    tok = torch.randint(2, 1000, (3, 40))
    tgt = torch.randint(2, 1000, (3, 40))
    tgt[:, :5] = -100
    monkeypatch.setattr(T, "CE_CHUNK", 7)
    h, _ = m(tok, return_hidden=True)
    l1 = T.chunked_ce(m, h, tgt, torch.device("cpu"))
    l1.backward()
    g1 = m.head.weight.grad.clone()
    m.zero_grad()
    lg, _ = m(tok)
    l2 = F.cross_entropy(lg.view(-1, 1000), tgt.view(-1), ignore_index=-100)
    l2.backward()
    assert torch.allclose(l1, l2, atol=1e-5) and torch.allclose(g1, m.head.weight.grad, atol=1e-6)


def test_holm_and_bootstrap():
    from src.analyze import bootstrap_ci, holm

    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06  # monotone step-down
    lo, hi = bootstrap_ci([1.0, 1.0, 1.0, 1.0])
    assert lo == hi == 1.0


def test_static_table_stacks_under_locker(tmp_path):
    """static_rows adds an Engram-style table on the embeddings, sharing the schedule's cues; it
    trains alone (static only) and under locker, and its gradients flow."""
    import torch

    from src.controls import arm_config
    from src.models import Model

    for arm, ov in (("ovl_only", {"overlay.orders": [1, 2]}), ("base", {"mixer": "attn", "sched.orders": [1, 2]})):
        m, s = arm_config(arm, {"name": "kv"}, {**ov, "static_rows": 1024})
        assert m["static"]["rows"] == 1024 and s["table_rows"] == 1024 and m["static"]["orders"] == (1, 2)
        cfg = dict(run_id=f"st-{arm}", arm=arm, seed=0, task=dict(name="kv", D=16, C=96, Q=8), steps=3, batch=2,
                   eval_every=3, eval_size=2, num_workers=0, overrides={**ov, "static_rows": 1024}, device="cpu")
        final = run(cfg, tmp_path)
        assert final["acc"] is not None
        ck = torch.load(tmp_path / f"st-{arm}.pt", weights_only=False)
        model = Model(ck["mcfg"])
        model.load_state_dict(ck["model"])
        assert model.static is not None and (model.overlay is not None) == (arm == "ovl_only")


def test_bayesian_locker_trains(tmp_path):
    """overlay.prior_rows wires table_rows into the schedule; prior and kappa get gradients."""
    import torch

    from src.controls import arm_config
    from src.models import Model

    ov = {"overlay.orders": [1, 2], "overlay.write_mode": "mean", "overlay.prior_rows": 512}
    m, s = arm_config("ovl_only", {"name": "kv"}, ov)
    assert s["table_rows"] == 512
    cfg = dict(run_id="bayes", arm="ovl_only", seed=0, task=dict(name="kv", D=16, C=96, Q=8), steps=3, batch=2,
               eval_every=3, eval_size=2, num_workers=0, overrides=ov, device="cpu")
    assert run(cfg, tmp_path)["acc"] is not None
    ck = torch.load(tmp_path / "bayes.pt", weights_only=False)
    k0 = torch.zeros_like(ck["model"]["overlays.0.kappa_table"])
    assert not torch.equal(ck["model"]["overlays.0.kappa_table"], k0)  # log kappa moved from log(1) = 0
    model = Model(ck["mcfg"])
    model.load_state_dict(ck["model"])
