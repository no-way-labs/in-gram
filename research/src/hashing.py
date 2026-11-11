"""N-gram hashing (plan.md §4.1).

addr(n, h, t) is computed over tokens[t-n+1 .. t]:
    x = 0
    for tok in window: x = (x * P[n][h] + tok + 1) mod (2**61 - 1)
    addr = x XOR SEED[n][h]

Vectorized form: x = sum_d (tok[t-d] + 1) * P^d  mod M,  d = 0..n-1.
Each term is reduced mod M (< 2^61) and the running sum is reduced after every term.
"""
from __future__ import annotations

import numpy as np

from .tasks import BOS, PAD, VOCAB

MOD = (1 << 61) - 1
MAX_ORDER = 32  # the vectorized sum is reduced mod M after every term, so any order fits in uint64
MAX_HEADS = 4

# PRIMES[n-1][h]: distinct primes (verified in tests).
PRIMES = [
    [1000003, 1000033, 1000037, 1000039],
    [1000081, 1000099, 1000117, 1000121],
    [1000133, 1000151, 1000159, 1000171],
    [1000183, 1000187, 1000193, 1000199],
    [1000211, 1000213, 1000231, 1000249],
    [1000253, 1000273, 1000289, 1000291],
    [1000303, 1000313, 1000333, 1000357],
    [1000367, 1000381, 1000393, 1000397],
]
# SEEDS[n-1][h]: fixed 61-bit constants.
SEEDS = [
    [0x0A3F1C5D2E7B9461, 0x1B5E7A2C4D8F0372, 0x05C9E1B3A7D24F86, 0x13D7A5F9C2E8B041],
    [0x0E2B8D4F6A1C3957, 0x17A4C6E8F0B2D193, 0x09F1E3D5B7A2C468, 0x1C6A8E0B2D4F7135],
    [0x04D8F2A6C0E3B597, 0x12B6D0F4A8C2E379, 0x0B7F3E1D9C5A2864, 0x18E2C4A6F8D0B153],
    [0x06A1C3E5F7B9D024, 0x15C7E9A1B3D5F682, 0x0D3B5F7A9C1E2846, 0x1A9D1F3B5E7C0428],
    [0x018BECA0A6119290, 0x04465037169D8977, 0x003E8EB03C2E374F, 0x1B30AC0E9E16A3F4],
    [0x0CFCAA5B5337CAFC, 0x08E9D3A9FF8115E9, 0x15704C92E1ADF5FD, 0x14C3EFE702F88991],
    [0x1CF7717506607CA4, 0x06E1DBE31CF95F0B, 0x1E56D1EC45CB8FAF, 0x057B16572325354A],
    [0x1283F5264F25E2EC, 0x0AD8E36E222506A8, 0x15C0313CE3E13350, 0x07BFFB8B00A6CE70],
]
SEEDS = [[s & MOD for s in row] for row in SEEDS]


def _extend_constants(max_order: int):
    """Rows for orders 9..max_order (added 2026-11-09 for long cues). Rows 1-8 are the original
    constants, untouched, so every existing address (and checkpoint) is unchanged. New primes are the
    next primes after the last listed one; new seeds come from a fixed-seed generator."""
    from sympy import nextprime

    p = PRIMES[-1][-1]
    rng = np.random.default_rng(20261109)
    while len(PRIMES) < max_order:
        row = []
        for _ in range(4):
            p = int(nextprime(p))
            row.append(p)
        PRIMES.append(row)
        SEEDS.append([int(rng.integers(0, 1 << 61, dtype=np.uint64)) & MOD for _ in range(4)])


_extend_constants(MAX_ORDER)

_TABLES: dict[tuple[int, int, int], np.ndarray] = {}


def _table(n: int, h: int, vocab: int = VOCAB) -> np.ndarray:
    """T[d, tok] = (tok + 1) * P^d mod M, as uint64; shape [n, vocab]. Exact: (tok+1) < 2^17 and
    P^d mod M < 2^61, so each product is reduced with Python ints."""
    key = (n, h, vocab)
    if key not in _TABLES:
        p = PRIMES[n - 1][h]
        T = np.zeros((n, vocab), dtype=np.uint64)
        for d in range(n):
            pd = pow(p, d, MOD)
            T[d] = np.array([((tok + 1) * pd) % MOD for tok in range(vocab)], dtype=np.uint64)
        _TABLES[key] = T
    return _TABLES[key]


def hash_ngram(window: list[int], n: int, h: int) -> int:
    """Scalar reference, literally as written in the spec."""
    assert len(window) == n
    x = 0
    for tok in window:
        x = (x * PRIMES[n - 1][h] + int(tok) + 1) % MOD
    return x ^ SEEDS[n - 1][h]


def pattern_offsets(order) -> tuple[int, ...]:
    """An order is an int n (contiguous N-gram: offsets 0..n-1 back from t) or an explicit tuple
    of offsets back from t, e.g. (0, 2, 3) = tokens t, t-2, t-3 (a skip-gram that drops t-1)."""
    if isinstance(order, (int, np.integer)):
        return tuple(range(int(order)))
    offs = tuple(sorted(int(o) for o in order))
    assert offs and offs[0] >= 0 and len(set(offs)) == len(offs), order
    return offs


def ngram_addresses(tokens: np.ndarray, n, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Address of the N-gram (or skip-gram pattern) ending at each position.

    tokens: int array [B, L] (position 0 is BOS). n: int order or tuple of offsets (pattern_offsets).
    x = sum_r (tok[t - o_r] + 1) * P^r over the pattern's sorted offsets o_r; a contiguous
    pattern reproduces the spec's hash exactly.
    Returns (addr int64 [B, L], valid bool [B, L]). Invalid where the pattern would reach
    before position 1 or touches PAD (or BOS).
    """
    offs = pattern_offsets(n)
    n_len, span = len(offs), offs[-1]
    assert 1 <= n_len <= MAX_ORDER and 0 <= h < MAX_HEADS
    tokens = np.asarray(tokens)
    B, L = tokens.shape
    vocab = VOCAB if tokens.max(initial=0) < VOCAB else 1 << int(tokens.max()).bit_length()
    T = _table(n_len, h, vocab)
    x = np.zeros((B, L), dtype=np.uint64)
    valid = np.zeros((B, L), dtype=bool)
    valid[:, span + 1 :] = True  # earliest token t - span >= 1
    for r, d in enumerate(offs):
        shifted = np.zeros_like(tokens)
        shifted[:, d:] = tokens[:, : L - d] if d else tokens
        x = (x + T[r][shifted]) % np.uint64(MOD)  # both terms < 2^61: the sum fits in uint64
        valid &= (shifted != PAD) & (shifted != BOS)
    addr = (x ^ np.uint64(SEEDS[n_len - 1][h])).astype(np.int64)
    return np.where(valid, addr, 0), valid
