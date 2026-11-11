import numpy as np
import torch

from ingram import InGram, InGramDecoder


def make(vocab=12, L=160, d_model=32, **kw):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tok = rng.integers(2, vocab, size=(2, L))
    tok[:, 0] = 1                      # a beginning-of-sequence token that no cue may contain
    layer = InGram(d_model, d_mem=8, table_rows=1 << 8, boundary_ids=(0, 1), pad_id=0, **kw).double()
    with torch.no_grad():              # larger than the initialization, so that the memory changes the output
        for p in (layer.W_v, layer.table, layer.W_K.weight, layer.W_O.weight):
            p.mul_(20)
    x = torch.randn(2, L, d_model, dtype=torch.float64)
    return layer, tok, x


def sequential(layer, tok, x):
    out = []
    for b in range(tok.shape[0]):
        dec = InGramDecoder(layer)
        out.append(torch.stack([dec.step(x[b, t], int(tok[b, t])) for t in range(tok.shape[1])]))
    return torch.stack(out)


def test_parallel_pass_equals_token_by_token_decoding():
    layer, tok, x = make()
    y = layer(x, layer.schedule(tok))
    assert (y - x).abs().mean() > 1e-3
    assert torch.allclose(y, sequential(layer, tok, x), atol=1e-10)


def test_static_table_has_no_slots():
    layer, tok, x = make(writes=False)
    y = layer(x, layer.schedule(tok))
    dec = InGramDecoder(layer)
    out = torch.stack([dec.step(x[0, t], int(tok[0, t])) for t in range(tok.shape[1])])
    assert torch.allclose(y[0], out, atol=1e-10) and dec.n_slots == 0


def test_causal():
    layer, tok, x = make()
    y = layer(x, layer.schedule(tok))
    tok2, x2 = tok.copy(), x.clone()
    tok2[:, 100:] = np.random.default_rng(1).integers(2, 12, size=tok2[:, 100:].shape)
    x2[:, 100:] = torch.randn_like(x2[:, 100:])
    y2 = layer(x2, layer.schedule(tok2))
    assert torch.allclose(y[:, :100], y2[:, :100], atol=1e-12)


def test_slot_count_and_bound():
    layer, tok, x = make()
    sched = layer.schedule(tok)
    dec = InGramDecoder(layer)
    for t in range(tok.shape[1]):
        dec.step(x[0, t], int(tok[0, t]))
    assert dec.n_slots == int(sched["n_slots"][0]) <= len(layer.cue_lengths) * tok.shape[1]


def test_gradients_reach_every_parameter():
    layer, tok, x = make()
    layer(x, layer.schedule(tok)).pow(2).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())
