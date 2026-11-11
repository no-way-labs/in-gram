"""Pieces for the Qwen3.8-Flash-Next-shaped stack: partial RoPE, static table at layer k
(reading chosen cue lengths only), hashed columns in chosen hgdn layers only."""
import numpy as np
import torch

from src.controls import arm_config
from src.models import CausalAttention, Model, apply_rope
from src.schedule import build_schedule
from src.train import sched_to_torch


def test_partial_rope_rotates_only_first_dims():
    torch.manual_seed(0)
    a = CausalAttention(32, 2, rope=True, rope_dims=4)
    x = torch.randn(1, 10, 32)
    q = torch.randn(1, 2, 10, 16)
    r = a.rope_dims
    out = torch.cat([apply_rope(q[..., :r]), q[..., r:]], -1)
    assert torch.equal(out[..., r:], q[..., r:]) and not torch.allclose(out[..., :r], q[..., :r])
    assert a(x).shape == x.shape


def _qfn_cfg(**extra):
    ov = {"d_model": 16, "heads": 2, "n_blocks": 4, "mixer": ["hgdn", "hgdn", "hgdn", "attn"],
          "rope": True, "pos_emb": False, "rope_dims": 4, "hash.orders": [1, 2, 3], "static_rows": 1 << 8,
          "static_orders": [2, 3], "static_at": 1, **extra}
    mcfg, scfg = arm_config("hgdn", dict(name="lm", L_max=24), ov)
    mcfg["vocab"] = 20
    return mcfg, scfg


def test_static_reads_selected_streams_at_layer():
    mcfg, scfg = _qfn_cfg()
    assert mcfg["static"]["streams"] == [1, 2] and mcfg["static"]["orders"] == (2, 3)
    model = Model(mcfg).double().eval()
    for blk in model.blocks[:3]:
        blk.mixer.force_naive = True
    tok = np.random.default_rng(0).integers(2, 8, size=(1, 24))
    s = sched_to_torch(build_schedule(tok, **scfg), "cpu")
    calls = []
    model.static.register_forward_hook(lambda m, i, o: calls.append(i[0].clone()))
    model(torch.as_tensor(tok), s)
    # the static table sees the output of block 1, not the embeddings
    emb = model.embed(torch.as_tensor(tok))
    assert len(calls) == 1 and not torch.allclose(calls[0], emb)


def test_hash_layers_limit_hashed_columns():
    mcfg, _ = _qfn_cfg(**{"hash.layers": [1]})
    model = Model(mcfg)
    orders = [blk.mixer.orders for blk in model.blocks[:3]]
    assert orders == [(), (1, 2, 3), ()]
