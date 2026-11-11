"""quant_eval: fake quantization is bounded and reversible, each mode changes what it claims to, and
generation / loop statistics behave on known inputs."""
import numpy as np
import torch

from src import overlay as overlay_mod
from src.controls import arm_config
from src.models import Model
from src.nll_dump import eval_batches, model_nll
from src.quant_eval import MODES, apply_mode, fq, gen_stats, generate, loop_flag, run

TASK = dict(k=2, key_alphabet=16, D=8, C=48, Q=8)


def _model():
    torch.manual_seed(0)
    mcfg, scfg = arm_config("ovl", TASK, {"mixer": "attn", "overlay.prior_rows": 64, "overlay.prior_add": True,
                                          "overlay.prior_init": False})
    m = Model(mcfg)
    for blk in m.blocks:
        blk.mixer.force_flash = False
    return m, mcfg, scfg


def test_fq_error_bound_and_levels():
    x = torch.randn(10, 128)
    for b in (8, 4):
        y = fq(x, b, 64)
        s = x.reshape(-1, 64).abs().amax(-1, keepdim=True) / (2 ** (b - 1) - 1)
        assert ((y - x).reshape(-1, 64).abs() <= s / 2 + 1e-6).all()
        assert all(len(torch.unique(r)) <= 2 ** b - 1 for r in y.reshape(-1, 64))


def test_each_mode_moves_nll_and_undo_restores_hooks():
    m, _, scfg = _model()
    with torch.no_grad():  # a random table is tiny; make it matter so tbl modes show up
        for o in m.modules():
            if isinstance(o, overlay_mod.Overlay):
                o.prior_table.normal_()
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    batches = eval_batches(TASK, 4, 4, scfg)
    dev = torch.device("cpu")
    from src.train import sched_to_torch

    def hid():  # hidden states: NLL is stored in float16, too coarse for small changes at random init
        b = batches[0]
        with torch.no_grad():
            return m(torch.as_tensor(b["tokens"]), sched_to_torch(b["sched"], dev), return_hidden=True)[0]

    base = hid()
    base_nll = np.nanmean(model_nll(m, batches, dev).astype(np.float32))
    scan0 = overlay_mod.segmented_scan
    for name in ("w4", "kv4", "tbl4", "st4", "local4"):
        m.load_state_dict(sd)
        undo = apply_mode(m, MODES[name])
        v = hid()
        undo()
        assert not torch.equal(v, base), name
        assert overlay_mod.segmented_scan is scan0
        assert all("qkv_heads" not in blk.mixer.__dict__ for blk in m.blocks)
    m.load_state_dict(sd)
    assert torch.equal(hid(), base)
    assert np.nanmean(model_nll(m, batches, dev).astype(np.float32)) == base_nll


def test_loop_flag_and_gen_stats():
    rng = np.random.default_rng(0)
    assert loop_flag(np.array(list(rng.integers(0, 1000, 200)) + [5, 6, 7] * 50))
    assert not loop_flag(rng.integers(0, 1000, 256))
    p = rng.integers(0, 1000, 100)
    s = gen_stats(p, p[10:60].copy(), p[10:60].copy())
    assert s["copy8"] == 1.0 and s["agree"] == 1.0 and s["exact_prefix"] == 50


def test_generate_and_run_smoke(tmp_path):
    m, mcfg, scfg = _model()
    prompts = eval_batches(TASK, 2, 2, scfg)[0]["tokens"][:, :20]
    g = generate(m, scfg, np.asarray(prompts), 5, 0.0, torch.device("cpu"))
    assert g.shape == (2, 5)
    path = tmp_path / "m.pt"
    torch.save(dict(model=m.state_dict(), mcfg=mcfg, scfg=scfg), path)
    out = run(dict(task=TASK, n=2, batch=2, modes=["bf16", "local"], gen_modes=["bf16"], prompt=20, new=4,
                   gen_batch=2, temps=(0.0,)), {"m": path})
    r = out["models"]["m"]
    assert set(r["nll"]) == {"bf16", "local"} and "bf16@T0" in r["gen"] and "reference" in r["gen"]
