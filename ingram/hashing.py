"""Addresses of N-gram cues.

The address of the cue that ends at position t is a polynomial hash of its N token ids,

    x = sum_{i=0}^{N-1} (w[t-i] + 1) * P_N**i   mod (2**61 - 1),      a = x XOR S_N,

with one prime P_N and one constant S_N per cue length (Section 2.1 of the paper). The slot of a cue is
keyed by this 61-bit address. The row of the learned table is a second hash of the address reduced modulo
the number of rows (`table_rows`).
"""
from __future__ import annotations

import numpy as np

MOD = (1 << 61) - 1
MAX_CUE_LENGTH = 32

# One prime and one 61-bit constant per cue length 1..32. These are the constants of the models in the paper.
PRIMES = [
    1000003, 1000081, 1000133, 1000183, 1000211, 1000253, 1000303, 1000367, 1000403, 1000429, 1000537,
    1000579, 1000621, 1000669, 1000721, 1000793, 1000859, 1000919, 1000973, 1001017, 1001069, 1001093,
    1001159, 1001197, 1001279, 1001321, 1001353, 1001389, 1001447, 1001501, 1001551, 1001593,
]
SEEDS = [
    0x0A3F1C5D2E7B9461, 0x0E2B8D4F6A1C3957, 0x04D8F2A6C0E3B597, 0x06A1C3E5F7B9D024, 0x018BECA0A6119290,
    0x0CFCAA5B5337CAFC, 0x1CF7717506607CA4, 0x1283F5264F25E2EC, 0x09FC9EB5F71FAB51, 0x1975279891BF42DE,
    0x0A119C723DF8DE8F, 0x1C2FAB75ECDC5FBB, 0x12BAB65E17CBFE23, 0x06D2B84EB02C714A, 0x03FEFA2DE02124D6,
    0x1500BC546378649D, 0x14FE2BFA2C50D362, 0x1CB581184225FB40, 0x0B7BE03821D3868C, 0x1DEC8CF286643808,
    0x19F06DB6E7B6BE5C, 0x10DF250A7698EA77, 0x16B8E29D1EE0E278, 0x08C26C90395805A4, 0x179BB2D1457A6ACE,
    0x13452ADA908ED43E, 0x065AFA4C48582986, 0x0DDA723589C36A19, 0x00997E037E3FE4B4, 0x1135E995B589CC54,
    0x1917F3D88FCCC17B, 0x1A37E27841441E8F,
]

_M = np.uint64(MOD)
_LOW29 = np.uint64((1 << 29) - 1)
_LOW32 = np.uint64(0xFFFFFFFF)


def _reduce(z: np.ndarray) -> np.ndarray:
    """z mod (2**61 - 1) for any uint64 z."""
    z = (z & _M) + (z >> np.uint64(61))
    return np.where(z >= _M, z - _M, z)


def _mulmod(a: np.ndarray, b: int) -> np.ndarray:
    """(a * b) mod (2**61 - 1), exact in uint64, for a < 2**32 and b < 2**61."""
    lo, hi = np.uint64(b & 0xFFFFFFFF), np.uint64(b >> 32)
    r = _reduce(a * hi)  # a * hi < 2**61
    # r * 2**32 mod M: with r = u * 2**29 + v, r * 2**32 = u * 2**61 + v * 2**32, and 2**61 = 1 (mod M)
    shifted = (r >> np.uint64(29)) + ((r & _LOW29) << np.uint64(32))
    return _reduce(_reduce(shifted) + _reduce(a * lo))


def hash_cue(window, n: int | None = None) -> int:
    """Address of one cue given as a sequence of token ids, oldest first (scalar reference)."""
    n = len(window) if n is None else n
    assert len(window) == n and 1 <= n <= MAX_CUE_LENGTH
    x = 0
    for tok in window:
        x = (x * PRIMES[n - 1] + int(tok) + 1) % MOD
    return x ^ SEEDS[n - 1]


def cue_addresses(tokens: np.ndarray, n: int, boundary_ids=()) -> tuple[np.ndarray, np.ndarray]:
    """Address of the cue of length n that ends at each position.

    tokens: integer array [B, L] with ids below 2**32 - 1. boundary_ids: token ids that no cue may contain
    (for example padding, a beginning-of-sequence token, or a separator between documents).
    Returns (address int64 [B, L], valid bool [B, L]). A position has no cue of length n when fewer than n
    tokens precede and include it, or when the cue would contain a boundary id; its address is 0.
    """
    assert 1 <= n <= MAX_CUE_LENGTH
    tokens = np.asarray(tokens)
    B, L = tokens.shape
    t64 = tokens.astype(np.uint64) + np.uint64(1)
    bad = np.isin(tokens, np.asarray(list(boundary_ids), dtype=tokens.dtype)) if len(boundary_ids) else np.zeros((B, L), bool)
    x = np.zeros((B, L), dtype=np.uint64)
    valid = np.zeros((B, L), dtype=bool)
    valid[:, n - 1:] = True
    p = PRIMES[n - 1]
    for i in range(n):
        if i >= L:
            break
        term = np.zeros((B, L), dtype=np.uint64)
        term[:, i:] = _mulmod(t64[:, : L - i], pow(p, i, MOD))
        x = _reduce(x + term)
        shifted_bad = np.zeros((B, L), dtype=bool)
        shifted_bad[:, i:] = bad[:, : L - i]
        valid &= ~shifted_bad
    addr = (x ^ np.uint64(SEEDS[n - 1])).astype(np.int64)
    return np.where(valid, addr, 0), valid


def mix64(addr) -> np.ndarray:
    """splitmix64 finalizer: a bijection on 64 bits that spreads the address over all of its bits."""
    z = np.asarray(addr).astype(np.uint64)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def table_row(addr, table_rows: int) -> np.ndarray:
    """Row of the learned table for an address."""
    return (mix64(addr) % np.uint64(table_rows)).astype(np.int64)
