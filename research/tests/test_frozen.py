"""Frozen backbone + bolt-on module (src/frozen.py), on a tiny stand-in with the HF layout."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.frozen import FrozenWithMemory, evaluate, make_module, n_trainable, train_frozen
from src.models import CausalAttention


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = CausalAttention(d, 2, force_flash=False)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return (x + self.mlp(self.n2(x)),)  # HF layers return tuples


class TinyLM(nn.Module):
    def __init__(self, vocab=40, d=32, n=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=d)
        self.model = nn.Module()
        self.model.embed = nn.Embedding(vocab, d)
        self.model.layers = nn.ModuleList(_Layer(d) for _ in range(n))
        self.lm_head = nn.Linear(d, vocab)

    def forward(self, input_ids, use_cache=False):
        x = self.model.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)[0]
        return SimpleNamespace(logits=self.lm_head(x))


CFG = dict(orders=(1, 2, 3), d_mem=16, steps=30, lr=3e-3, warmup=5)


def _tok(B, L, seed, vocab=40):
    rng = np.random.default_rng(seed)
    return rng.integers(2, vocab, size=(B, L))


def test_only_module_trains_and_none_is_identity():
    torch.manual_seed(0)
    base = TinyLM()
    plain = FrozenWithMemory(base, None)
    assert n_trainable(plain) == 0
    t = torch.as_tensor(_tok(1, 20, 0))
    ref = base(t).logits
    assert torch.equal(plain(t), ref)
    m = FrozenWithMemory(TinyLM(), "edm", insert_at=(1,), cfg=CFG)
    assert all(not p.requires_grad for p in m.base.parameters())
    assert n_trainable(m) == sum(p.numel() for p in m.mods.parameters())


def test_adapter_matches_edm_params_and_starts_as_identity():
    d = 64
    e, a = make_module("edm", d, CFG), make_module("adapter", d, CFG)
    pe, pa = n_trainable(e), n_trainable(a)
    assert abs(pa - pe) / pe < 0.05, (pa, pe)
    torch.manual_seed(0)
    base = TinyLM()
    m = FrozenWithMemory(base, "adapter", insert_at=(1,), cfg=CFG)
    t = torch.as_tensor(_tok(1, 20, 1))
    assert torch.allclose(m(t), base(t).logits)


def test_edm_hook_is_causal():
    from src.frozen import _sched

    torch.manual_seed(0)
    m = FrozenWithMemory(TinyLM(), "edm", insert_at=(1, 2), cfg=CFG).eval()
    a = _tok(1, 40, 2)
    b = a.copy()
    b[0, 25:] = _tok(1, 15, 3)
    la = m(torch.as_tensor(a), _sched(a, CFG, torch.device("cpu")))
    lb = m(torch.as_tensor(b), _sched(b, CFG, torch.device("cpu")))
    assert torch.allclose(la[:, :25], lb[:, :25], atol=1e-5)
    assert not torch.allclose(la[:, 25:], lb[:, 25:])


def test_training_changes_only_module_and_reduces_loss_on_repeats():
    torch.manual_seed(0)
    base = TinyLM()
    before = {k: v.clone() for k, v in base.state_dict().items()}
    m = FrozenWithMemory(base, "edm", insert_at=(1,), cfg=CFG)
    rng = np.random.default_rng(0)

    def batches():
        while True:  # each sequence repeats its first half: copying lowers the loss
            h = rng.integers(2, 40, size=(4, 24))
            yield np.concatenate([h, h], 1)

    dev = torch.device("cpu")
    w = [next(batches()) [0] for _ in range(4)]
    labels = [np.r_[np.zeros(24, int), np.full(24, 2)] for _ in w]
    l0 = evaluate(m, w, labels, CFG, dev)
    train_frozen(m, batches(), dict(CFG, steps=60), dev)
    l1 = evaluate(m, w, labels, CFG, dev)
    for k, v in base.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert l1["by_regime"]["long"] < l0["by_regime"]["long"]
    assert l0["by_regime"]["short"] is None
    assert abs(sum(l1["frac_regime"].values()) - 1) < 1e-9
    import json

    json.dumps(l1)  # logged as JSON on the host (numpy floats broke the first GPU smoke run)


def test_model_ids_unshift_and_sep():
    from src.frozen import model_ids

    w = np.array([[1, 2, 5, 40, 1, 7]])
    out = model_ids(w, dict(id_offset=2, bos_id=1))
    assert out.tolist() == [[1, 0, 3, 38, 1, 5]]
    assert model_ids(w, {}) is w
