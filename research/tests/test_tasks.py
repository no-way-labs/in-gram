import numpy as np
import pytest

from src.tasks import (BOS, DIGIT0, IGNORE, KEY0, NOISE0, PAD, PT_QFINAL, VAL0, KVConfig, ModSumConfig,
                       kv_batch, modsum_batch)


def _context_bindings(tok, cfg):
    """Scan context; return dict key-tuple -> list of (value_pos, value)."""
    k, C = cfg.k, cfg.C
    out = {}
    t = 1
    while t <= C:
        if KEY0 <= tok[t] < VAL0:
            key = tuple(tok[t : t + k])
            out.setdefault(key, []).append((t + k, tok[t + k]))
            t += k + 1
        else:
            assert NOISE0 <= tok[t] < NOISE0 + 128
            t += 1
    return out


@pytest.mark.parametrize("rebind,corrupt", [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 1.0)])
def test_kv_structure(rebind, corrupt):
    cfg = KVConfig(D=48, C=384, Q=32, rebind=rebind, corrupt=corrupt)
    batch = kv_batch(cfg, 8, np.random.default_rng(3))
    for b in range(8):
        tok = batch["tokens"][b]
        assert tok[0] == BOS
        L_used = 1 + cfg.C + cfg.Q * (cfg.k + 1)
        assert (tok[L_used:] == PAD).all() and (tok[:L_used] != PAD).all()
        binds = _context_bindings(tok, cfg)
        assert len(binds) == cfg.D  # keys distinct
        n_reb = sum(len(v) == 2 for v in binds.values())
        assert n_reb == round(rebind * cfg.D)
        scored = np.nonzero(batch["targets"][b] != IGNORE)[0]
        assert len(scored) == cfg.Q
        assert batch["corrupt"][b].sum() == round(corrupt * cfg.Q)
        for s in scored:
            assert batch["pos_type"][b, s] == PT_QFINAL
            key = tuple(tok[s - cfg.k + 1 : s + 1])
            target = batch["targets"][b, s]
            assert tok[s + 1] == target  # teacher-forced answer token follows
            if batch["corrupt"][b, s]:
                assert key not in binds
                assert sum(x != y for x, y in zip(key, _orig_key(binds, batch["src_pos"][b, s], cfg))) == 1
            else:
                pos, val = max(binds[key])  # latest binding
                assert target == val and batch["src_pos"][b, s] == pos
            if rebind:
                assert target == tok[batch["src_pos"][b, s]]


def _orig_key(binds, src_pos, cfg):
    for key, lst in binds.items():
        if any(p == src_pos for p, _ in lst):
            return key
    raise AssertionError


def test_kv_rejects_bad_config():
    with pytest.raises(ValueError):
        kv_batch(KVConfig(D=200, C=384), 1, np.random.default_rng(0))


def test_modsum():
    cfg = ModSumConfig()
    batch = modsum_batch(cfg, 4, np.random.default_rng(0))
    tok, tgt = batch["tokens"], batch["targets"]
    assert tok.shape == (4, 512)
    for b in range(4):
        d = tok[b, 1 : 2 * cfg.n_digits + 1 : 2] - DIGIT0
        s = np.cumsum(d) % 10
        assert (tgt[b, 1 : 2 * cfg.n_digits + 1 : 2] - DIGIT0 == s).all()
        assert (tok[b, 2 * cfg.n_digits + 1 :] == PAD).all()


def test_topic_kv_structure():
    from src.tasks import TOPIC0, TopicKVConfig, topic_kv_batch

    cfg = TopicKVConfig(D=16, C_t=96, Q=16, L_max=384)
    b = topic_kv_batch(cfg, 3, np.random.default_rng(0))
    for i in range(3):
        tok = b["tokens"][i]
        for s in np.nonzero(b["targets"][i] != IGNORE)[0]:
            t = tok[s - cfg.k] - TOPIC0
            base = int(np.nonzero(tok[: 1 + cfg.T * (1 + cfg.C_t)] == TOPIC0 + t)[0][0])  # its section
            assert base < b["src_pos"][i, s] <= base + cfg.C_t  # answer comes from the queried topic
            assert tok[b["src_pos"][i, s]] == b["targets"][i, s]
            other = [tok[p] for p in range(cfg.k + 1, cfg.T * (1 + cfg.C_t) + 1)
                     if tuple(tok[p - cfg.k:p]) == tuple(tok[s - cfg.k + 1:s + 1])]
            assert len(set(other)) == cfg.T  # same key bound to T different values: a homograph


def test_topic_order_shuffles():
    from src.tasks import TOPIC0, TopicKVConfig, topic_kv_batch

    cfg = TopicKVConfig(D=8, C_t=48, Q=8, L_max=192)
    b = topic_kv_batch(cfg, 40, np.random.default_rng(0))
    firsts = {int(b["tokens"][i, 1]) - TOPIC0 for i in range(40)}
    assert firsts == {0, 1}  # either topic can come first


@pytest.mark.parametrize("rebinds", [2, 3])
def test_kv_multi_rebind(rebinds):
    cfg = KVConfig(D=32, C=480, Q=32, L_max=600, rebind=0.5, rebinds=rebinds)
    batch = kv_batch(cfg, 8, np.random.default_rng(5))
    for b in range(8):
        tok = batch["tokens"][b]
        binds = _context_bindings(tok, cfg)
        assert len(binds) == cfg.D
        counts = sorted(len(v) for v in binds.values())
        assert counts.count(rebinds + 1) == 16 and counts.count(1) == 16
        for lst in binds.values():
            assert len({v for _, v in lst}) == len(lst)  # all values of a key distinct
        for s in np.nonzero(batch["targets"][b] != IGNORE)[0]:
            key = tuple(tok[s - cfg.k + 1 : s + 1])
            pos, val = max(binds[key])
            assert batch["targets"][b, s] == val and batch["src_pos"][b, s] == pos

