"""Word-key passkey and multi-key retrieval evals (src/longeval.py): prompt layout, scoring
positions, metrics, and end-to-end runs on tiny CPU models. No downloads: the filler is synthetic
text (GPT-2 BPE via tiktoken) injected in place of WikiText-103."""
import numpy as np
import pytest
import torch

from src import longeval, text
from src.longeval import WORD_POOL, _encode, _word_ids, multikey_eval, wordkey_eval
from src.tasks import BOS


@pytest.fixture(scope="module")
def filler():
    """~6k tokens of sentences that use pool words (to exercise scrubbing) and end in ' .'."""
    rng = np.random.default_rng(0)
    extra = "the of a was in his first new year city time film game".split()
    sents = [" " + " ".join(rng.choice(WORD_POOL + extra * 4, size=int(rng.integers(5, 15)))) + " ."
             for _ in range(600)]
    return _encode("".join(sents)).astype(np.uint16)


class _Recorder:
    """Stands in for _nll_at: records the sequence and returns perfect scores."""

    def __init__(self):
        self.calls = []

    def __call__(self, model, scfg, tokens, pos, device, cands=None):
        self.calls.append(dict(tokens=tokens.copy(), pos=pos.copy(), cands=cands))
        k = len(pos)
        info = dict(sec=0.0, peak_gb=None, n_slots=0, top1=np.ones(k, bool), rank=np.ones(k, np.int64))
        if cands is not None:
            info["cand_top1"] = np.ones(k, bool)
        return np.zeros(k), info


def test_word_pool_single_common_tokens():
    ids = _word_ids()
    assert len(ids) == len(WORD_POOL) == len(set(ids.tolist())) >= 32  # 16 pairs need 32 words
    assert len(_encode(" " + longeval.SCRUB_WORD)) == 1 and longeval.SCRUB_WORD not in WORD_POOL


@pytest.mark.parametrize("L", [128, 512])
def test_wordkey_prompt_and_scoring(monkeypatch, filler, L):
    rec = _Recorder()
    monkeypatch.setattr(longeval, "_nll_at", rec)
    r = wordkey_eval(None, {}, torch.device("cpu"), L, 3, filler, np.random.default_rng(1), n_words=3)
    assert r["acc"] == 1.0 and r["rank_mean"] == 1.0 and r["top1_by_pos"] == [1.0] * 3 and r["words"] == 3
    assert r["scrubbed"] > 0  # the synthetic filler is full of pool words
    ids = set(_word_ids().tolist())
    for c in rec.calls:
        t, pos = c["tokens"], c["pos"]
        assert len(t) == L and t[0] == BOS
        ans = t[pos + 1]
        assert len(ans) == 3 and set(ans.tolist()) <= ids and len(set(ans.tolist())) == 3
        assert np.array_equal(pos, np.arange(L - 4, L - 1))  # the last 3 tokens are scored
        q = _encode(" The pass key is")
        assert np.array_equal(t[L - 3 - len(q):L - 3], q)
        for w in ans:  # twice in the needle, once as the answer, never in the filler
            assert (t == w).sum() == 3
        assert not (np.isin(t, list(ids)).sum() - 9)  # no other pool word anywhere


@pytest.mark.parametrize("pairs", [4, 16])
def test_multikey_prompt_and_scoring(monkeypatch, filler, pairs):
    rec = _Recorder()
    monkeypatch.setattr(longeval, "_nll_at", rec)
    L = 1024
    r = multikey_eval(None, {}, torch.device("cpu"), L, 4, filler, np.random.default_rng(2), pairs=pairs)
    assert r["acc"] == r["acc_choice"] == 1.0 and r["chance_choice"] == 1 / pairs and len(r["per_trial"]) == 4
    is_ = int(_encode(" is")[0])
    for c, tr in zip(rec.calls, r["per_trial"]):
        t, pos, cands = c["tokens"], c["pos"], c["cands"]
        assert len(t) == L and t[0] == BOS and len(pos) == 1 and pos[0] == L - 2
        val = int(t[-1])
        assert t[-2] == is_ and val in cands.tolist() and len(set(cands.tolist())) == pairs
        key = int(t[-3])
        assert (t == key).sum() == 2 and (t == val).sum() == 2  # its needle + the query / answer
        # Each key is followed by " is <value>" in exactly one needle; keys never co-occur in filler.
        pool = _word_ids()
        words = t[np.isin(t, pool)]
        assert len(words) == 2 * pairs + 2 and len(set(words.tolist())) == 2 * pairs
        k_at = np.flatnonzero(t[:-3] == key)[0]
        assert t[k_at + 1] == is_ and t[k_at + 2] == val  # the queried needle binds key -> value
        assert abs(tr["depth"] - (k_at - 4) / L) < 1e-3  # needle " The special word for KEY": KEY at +4
        assert 0 < tr["depth"] < 1


def test_rank_and_candidates_match_logits():
    from src.models import Model

    torch.manual_seed(0)
    m = Model(dict(mixer="attn", d_model=16, n_blocks=1, heads=2, mlp_hidden=32, max_len=64, vocab=300)).eval()
    tokens = np.concatenate([[BOS], np.random.default_rng(0).integers(2, 300, size=40)]).astype(np.int64)
    pos = np.array([5, 17, 39])
    cands = np.array([7, 8, 9, int(tokens[18])])
    nll, info = longeval._nll_at(m, {}, tokens, pos, torch.device("cpu"), cands=cands)
    with torch.no_grad():
        lg = m(torch.as_tensor(tokens[None]))[0][0, pos].float()
    tg = torch.as_tensor(tokens[pos + 1])
    lt = lg.gather(1, tg[:, None])
    assert np.array_equal(info["rank"], ((lg > lt).sum(-1) + 1).numpy())
    assert np.allclose(nll, torch.nn.functional.cross_entropy(lg, tg, reduction="none").numpy(), atol=1e-5)
    assert np.array_equal(info["top1"], info["rank"] == 1)
    assert info["cand_top1"][1] == bool((lg[1, torch.as_tensor(cands)].max() <= lt[1, 0]))


@pytest.mark.parametrize("arm", ["attn", "ovl_only"])
def test_long_eval_end_to_end(monkeypatch, filler, arm):
    from src.controls import arm_config
    from src.models import Model

    monkeypatch.setattr(text, "load_tokens", lambda cfg="", split="train": filler)
    mcfg, scfg = arm_config(arm, {"name": "lm", "L_max": 512},
                            {"d_model": 16, "n_blocks": 1, "heads": 2, "mlp_hidden": 32, "vocab": text.vocab_size(),
                             **({"overlay.orders": [1, 2], "overlay.H": 1} if arm == "ovl_only" else {})})
    torch.manual_seed(0)
    model = Model(mcfg)
    spec = dict(wordkey_lengths=[256, 128], wordkey_n=2, wordkey_words=2,
                multikey_lengths=[512], multikey_n=2, multikey_pairs=[2, 4])
    out = longeval.long_eval(model, scfg if mcfg.get("overlay") else {}, torch.device("cpu"), spec)
    assert [r["L"] for r in out["wordkey"]] == [128, 256] and all("error" not in r for r in out["wordkey"])
    assert [(r["L"], r["pairs"]) for r in out["multikey"]] == [(512, 2), (512, 4)]
    V = text.vocab_size()
    for r in out["wordkey"] + out["multikey"]:
        assert "error" not in r, r.get("error")
        assert 0 <= r["acc"] <= 1 and 1 <= r["rank_mean"] <= V and r["nll_tok"] > 0
    assert len(out["wordkey"][0]["rank_by_pos"]) == 2
    assert out["passkey"] == [] and out["probe"] == []
    import json

    json.dumps(out)  # results travel as JSON log lines (remote.py)
    # Deterministic: same seed, same prompts and scores.
    again = longeval.long_eval(model, scfg if mcfg.get("overlay") else {}, torch.device("cpu"), spec)
    assert again["multikey"][1]["per_trial"] == out["multikey"][1]["per_trial"]
