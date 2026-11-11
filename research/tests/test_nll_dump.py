"""nll_dump: per-token NLL from a saved checkpoint matches train.evaluate's mean loss on the same windows."""
import numpy as np
import torch

from src.nll_dump import eval_batches, model_nll
from src.controls import arm_config
from src.models import Model
from src.train import evaluate


def test_nll_dump_mean_matches_evaluate():
    torch.manual_seed(0)
    task = dict(k=2, key_alphabet=16, D=8, C=48, Q=8)
    mcfg, scfg = arm_config("ovl", task, {"mixer": "attn"})
    model = Model(mcfg)
    batches = eval_batches(task, 8, 4, scfg)
    nll = model_nll(model, batches, torch.device("cpu"))
    ev = evaluate(model, batches, torch.device("cpu"), detail=False)
    assert abs(float(np.nanmean(nll.astype(np.float32))) - ev["loss"]) < 1e-2


def test_attn_probs_reproduce_layer_output_and_stats_are_masses():
    from src.models import CausalAttention
    from src.nll_dump import attn_probs, attn_stats

    torch.manual_seed(1)
    m = CausalAttention(16, 2, conv=4, rope=True, force_flash=False)
    h = torch.randn(1, 40, 16)
    with torch.no_grad():
        p = attn_probs(m, h, slice(0, 40))
        _, _, v = m.qkv_heads(h)
        out = m.out((p @ v).transpose(1, 2).reshape(1, 40, 16))
        assert torch.allclose(out, m(h), atol=1e-5)

    task = dict(k=2, key_alphabet=16, D=8, C=48, Q=8)
    mcfg, scfg = arm_config("base", task, {"mixer": ["attn", "gdn"][:1] * 2, "n_blocks": 2, "rope": True,
                                           "pos_emb": False})
    model = Model(mcfg)
    for blk in model.blocks:
        blk.mixer.force_flash = False
    batches = eval_batches(task, 4, 4, scfg)
    st = attn_stats(model, batches, torch.device("cpu"), far=8, chunk=16)
    assert set(st) == {"0", "1"}
    for v in st.values():
        for h_ in range(2):
            assert -1e-6 <= v["far_ind"][h_] <= v["far"][h_] + 1e-6 <= 1 + 1e-5
            assert v["far_copy"][h_] <= v["far"][h_] + 1e-6
