import numpy as np
import pytest
from sympy import isprime

from src.hashing import MOD, PRIMES, SEEDS, hash_ngram, ngram_addresses
from src.tasks import BOS, PAD


def test_constants_distinct_primes():
    flat = [p for row in PRIMES for p in row]
    assert len(set(flat)) == len(flat) and all(isprime(p) for p in flat)
    seeds = [s for row in SEEDS for s in row]
    assert len(set(seeds)) == len(seeds) and all(0 <= s <= MOD for s in seeds)


def test_vectorized_matches_spec_loop():
    rng = np.random.default_rng(0)
    toks = rng.integers(2, 320, size=(3, 40))
    toks[:, 0] = BOS
    for n in (1, 2, 3, 4, 8, 9, 16, 32):
        for h in range(4):
            addr, valid = ngram_addresses(toks, n, h)
            for b in range(3):
                for t in range(n, 40):
                    assert valid[b, t]
                    assert addr[b, t] == hash_ngram(list(toks[b, t - n + 1 : t + 1]), n, h)


def test_deterministic():
    toks = np.random.default_rng(1).integers(2, 320, size=(2, 30))
    a1, v1 = ngram_addresses(toks, 2, 0)
    a2, v2 = ngram_addresses(toks.copy(), 2, 0)
    assert (a1 == a2).all() and (v1 == v2).all()


def test_orders_and_heads_differ():
    toks = np.random.default_rng(2).integers(2, 320, size=(4, 64))
    toks[:, 0] = BOS
    addrs = {}
    for n in (1, 2, 3):
        for h in range(2):
            a, v = ngram_addresses(toks, n, h)
            addrs[n, h] = a[:, 3:]
    keys = list(addrs)
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            assert (addrs[k1] != addrs[k2]).mean() > 0.99


def test_bos_and_pad_get_no_address():
    toks = np.array([[BOS, 10, 11, 12, PAD, PAD], [BOS, 20, PAD, 21, 22, 23]])
    _, v1 = ngram_addresses(toks, 1, 0)
    _, v2 = ngram_addresses(toks, 2, 0)
    assert v1.tolist() == [[0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 1, 1]]
    assert v2.tolist() == [[0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]


def test_skipgram_patterns():
    from src.hashing import pattern_offsets

    toks = np.random.default_rng(3).integers(2, 300, size=(2, 30))
    toks[:, 0] = BOS
    a4, v4 = ngram_addresses(toks, 4, 1)
    p4, w4 = ngram_addresses(toks, (0, 1, 2, 3), 1)
    assert (a4 == p4).all() and (v4 == w4).all()  # contiguous pattern == spec order
    a3, v3 = ngram_addresses(toks, 3, 0)
    s3, w3 = ngram_addresses(toks, (1, 2, 3), 0)  # drop the current token
    assert (s3[:, 1:] == a3[:, :-1]).all() and (w3[:, 1:] == v3[:, :-1]).all()
    g, vg = ngram_addresses(toks, (0, 2, 3), 1)  # drop t-1: changing t-1 must not change it
    t2 = toks.copy()
    t2[:, 10] = 299 - t2[:, 10] + 2
    g2, _ = ngram_addresses(t2, (0, 2, 3), 1)
    assert g[0, 11] == g2[0, 11] and g[0, 10] != g2[0, 10]
    assert pattern_offsets(3) == (0, 1, 2) and pattern_offsets([3, 0, 2]) == (0, 2, 3)


@pytest.mark.parametrize("n", [5, 6, 8])
def test_long_orders_match_scalar_reference(n):
    """Orders up to 8 (long cues for million-token contexts) match the spec's scalar hash, with
    50k-vocab ids so the uint64 sum of n terms is exercised near its bound."""
    rng = np.random.default_rng(n)
    toks = rng.integers(2, 50259, size=(2, 64))
    for h in range(4):
        addr, valid = ngram_addresses(toks, n, h)
        for b in range(2):
            for t in range(n, 64):
                assert valid[b, t] and addr[b, t] == hash_ngram(list(toks[b, t - n + 1 : t + 1]), n, h)


def test_original_constants_unchanged():
    """Orders 1-8 keep their 2026 constants (existing checkpoints depend on the addresses)."""
    assert PRIMES[0][0] == 1000003 and PRIMES[7][3] == 1000397
    assert SEEDS[0][0] == 0x0A3F1C5D2E7B9461 & MOD and SEEDS[7][3] == 0x07BFFB8B00A6CE70 & MOD
    assert len(PRIMES) == len(SEEDS) >= 32
