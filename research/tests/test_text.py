"""Real-text task: no downloads (a fake token stream is injected)."""
import json

import numpy as np
import pytest

from src import text
from src.hashing import hash_ngram, ngram_addresses
from src.tasks import BOS, IGNORE


@pytest.fixture
def fake_stream(monkeypatch):
    rng = np.random.default_rng(0)
    stream = (rng.integers(0, 50, size=20_000) + text.SHIFT).astype(np.uint16)  # small vocab -> repeats
    monkeypatch.setitem(text._STREAMS, ("fake", "train"), stream)
    monkeypatch.setitem(text._STREAMS, ("fake", "validation"), stream[::-1].copy())
    return stream


def test_repeat_distance():
    toks = np.array([[BOS, 5, 6, 7, 5, 6, 9, 5, 6]])
    d = text.repeat_distance(toks, 2)
    assert d[0, 5] == 3 and d[0, 8] == 3 and d[0, 2] == -1 and d[0, 7] == -1


def test_repeat_mask():
    toks = np.array([[BOS, 5, 6, 7, 5, 6, 9, 5, 6]])
    m = text.repeat_mask(toks, 2)
    # bigram (5,6) first ends at 2; repeats end at 5 and 8
    assert np.nonzero(m[0])[0].tolist() == [5, 8]


def test_lm_batch_alignment(fake_stream):
    cfg = text.LMConfig(dataset="fake", L_max=64)
    b = text.lm_batch(cfg, 3, np.random.default_rng(1))
    tok, tgt = b["tokens"], b["targets"]
    assert tok.shape == (3, 64) and (tok[:, 0] == BOS).all() and (tgt[:, 0] == IGNORE).all()
    assert (tgt[:, 1:-1] == tok[:, 2:]).all()  # next-token targets
    assert set(np.unique(b["pos_type"][:, 1:])) <= {text.PT_TEXT_NOVEL, text.PT_TEXT_REP_NEAR,
                                                    text.PT_TEXT_REP_MID, text.PT_TEXT_REP_FAR}


def test_hashing_large_vocab_matches_spec():
    toks = np.array([[BOS, 50000, 50001, 50000, 50001, 300]])
    for n in (1, 2, 3):
        a, v = ngram_addresses(toks, n, 1)
        for t in range(n, 6):
            assert a[0, t] == hash_ngram(list(toks[0, t - n + 1 : t + 1]), n, 1)


def test_canonical_map_merges_case_and_space():
    canon = text.canonical_map()
    enc = text._encoder()
    ids = [enc.encode_single_token(s) + text.SHIFT for s in ("Paris", " Paris", " paris")
           if len(enc.encode(s)) == 1]
    assert len(ids) >= 2 and len({int(canon[i]) for i in ids}) == 1
    assert canon[0] == 0 and canon[1] == 1  # PAD / BOS untouched


def test_lm_training_end_to_end(tmp_path, fake_stream, monkeypatch):
    from src.train import run

    monkeypatch.setattr(text, "vocab_size", lambda: 50 + text.SHIFT + 14)
    cfg = dict(run_id="lm", arm="ovl", task=dict(name="lm", dataset="fake", L_max=64), steps=2, eval_every=2,
               eval_size=8, batch=4, num_workers=0, overrides={"mixer": "attn"})
    final = run(cfg, tmp_path)
    assert final["acc"] is not None
    rec = [json.loads(l) for l in (tmp_path / "lm.jsonl").read_text().splitlines() if '"step"' in l][0]
    assert {"text_novel", "text_rep_near"} <= set(rec["loss_by_type"])


def test_distance_bins(fake_stream):
    cfg = text.LMConfig(dataset="fake", L_max=2048)
    b = text.lm_batch(cfg, 2, np.random.default_rng(0))
    d = text.repeat_distance(b["tokens"], 2)
    pt = b["pos_type"]
    assert (pt[(d >= 1024)] == text.PT_TEXT_REP_FAR).all()
    assert (pt[(d > 0) & (d < 256)] == text.PT_TEXT_REP_NEAR).all()
