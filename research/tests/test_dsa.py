"""DSA-style mixer (src/models.py DSAMixer): brute-force check, dense limit, causality, indexer loss."""
import math

import torch
import torch.nn.functional as F

from src.models import DSAMixer, Model


def _brute(m, x):
    B, L, D = x.shape
    H, dh = m.H, m.dh
    q, k, v = m.qkv(x).view(B, L, 3, H, dh).unbind(2)
    q, k = m.q_norm(q), m.k_norm(k)
    out = torch.zeros(B, L, D, dtype=x.dtype)
    for b in range(B):
        qi = m.iq(x[b]).view(L, m.HI, m.DI)
        ki = m.ik(x[b])
        wi = m.iw(x[b])
        for t in range(L):
            I = torch.stack([(wi[t] * F.relu((qi[t] * ki[s]).sum(-1) / math.sqrt(m.DI))).sum() for s in range(t + 1)])
            sel = I.topk(min(m.topk, t + 1)).indices
            for h in range(H):
                sc = (k[b, sel, h] * q[b, t, h]).sum(-1) / math.sqrt(dh)
                out[b, t, h * dh:(h + 1) * dh] = (torch.softmax(sc, 0).unsqueeze(-1) * v[b, sel, h]).sum(0)
    return m.out(out * torch.sigmoid(m.gate(x)))


def test_dsa_matches_brute_force():
    torch.manual_seed(0)
    m = DSAMixer(16, 2, topk=5, idx_heads=3, idx_dim=4, chunk=7).double()
    x = torch.randn(2, 20, 16, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(m(x), _brute(m, x), atol=1e-10)


def test_dsa_dense_limit_and_causality():
    torch.manual_seed(1)
    m = DSAMixer(16, 2, topk=64, chunk=8).double().eval()
    x = torch.randn(1, 24, 16, dtype=torch.float64)
    with torch.no_grad():
        y = m(x)
        q, k, v = m.qkv(x).view(1, 24, 3, 2, 8).permute(2, 0, 3, 1, 4)
        q, k = m.q_norm(q), m.k_norm(k)
        dense = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(1, 24, 16)
        assert torch.allclose(y, m.out(dense * torch.sigmoid(m.gate(x))), atol=1e-10)
        m.topk = 4
        x2 = x.clone()
        x2[:, 15:] += 1.0
        assert torch.allclose(m(x)[:, :15], m(x2)[:, :15], atol=1e-12)


def test_dsa_indexer_loss_trains_only_the_indexer():
    torch.manual_seed(2)
    m = DSAMixer(16, 2, topk=4, chunk=8).train()
    x = torch.randn(2, 16, 16, requires_grad=True)
    m(x)
    assert m.aux_loss is not None and m.aux_loss.item() >= 0
    m.aux_loss.backward()
    assert all(p.grad is not None for p in (m.iq.weight, m.ik.weight, m.iw.weight))
    assert m.qkv.weight.grad is None and x.grad is None


def test_dsa_model_has_no_position_embedding():
    model = Model(dict(mixer=["attn", "dsa"], d_model=16, n_blocks=2, heads=2, mlp_hidden=32, max_len=32,
                       dsa_topk=4, pos_emb=False))
    assert model.pos is None
    model2 = Model(dict(mixer=["dsa", "dsa"], d_model=16, n_blocks=2, heads=2, mlp_hidden=32, max_len=32))
    assert model2.pos is None
    out, _ = model2(torch.randint(0, 50, (1, 12)), return_hidden=True)
    assert out.shape == (1, 12, 16)


def test_swa_matches_banded_brute_force():
    from src.models import CausalAttention

    torch.manual_seed(3)
    m = CausalAttention(16, 2, window=5, force_flash=False).double()
    x = torch.randn(1, 20, 16, dtype=torch.float64)
    with torch.no_grad():
        y = m(x)
        q, k, v = m.qkv(x).view(1, 20, 3, 2, 8).permute(2, 0, 3, 1, 4)
        ref = torch.zeros(1, 2, 20, 8, dtype=torch.float64)
        for t in range(20):
            lo = max(0, t - 4)
            sc = (q[0, :, t:t + 1] @ k[0, :, lo:t + 1].transpose(-1, -2)) / 8 ** 0.5
            ref[0, :, t] = (torch.softmax(sc, -1) @ v[0, :, lo:t + 1])[:, 0]
        assert torch.allclose(y, m.out(ref.transpose(1, 2).reshape(1, 20, 16)), atol=1e-10)
