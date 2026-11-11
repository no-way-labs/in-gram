import numpy as np

from ingram.hashing import MOD, PRIMES, SEEDS, cue_addresses, hash_cue, table_row


def test_constants():
    assert len(PRIMES) == len(SEEDS) == 32 and len(set(PRIMES)) == 32
    assert all(0 <= s < MOD for s in SEEDS)


def test_vectorized_matches_scalar():
    rng = np.random.default_rng(0)
    for vocab in (5, 1000, 200_000):
        tok = rng.integers(0, vocab, size=(2, 80))
        for n in (1, 2, 4, 8, 16, 32):
            addr, valid = cue_addresses(tok, n)
            assert not valid[:, : n - 1].any() and valid[:, n - 1:].all()
            for b in range(2):
                for t in range(n - 1, 80, 7):
                    assert int(addr[b, t]) == hash_cue(tok[b, t - n + 1: t + 1].tolist(), n)


def test_boundary_ids_break_cues():
    tok = np.array([[5, 6, 7, 1, 8, 9, 10, 11]])
    _, valid = cue_addresses(tok, 2, boundary_ids=(1,))
    assert valid[0].tolist() == [False, True, True, False, False, True, True, True]


def test_same_cue_same_address_and_row():
    tok = np.array([[3, 4, 9, 3, 4, 2]])
    addr, _ = cue_addresses(tok, 2)
    assert addr[0, 1] == addr[0, 4] and addr[0, 1] != addr[0, 2]
    rows = table_row(addr, 1 << 16)
    assert rows[0, 1] == rows[0, 4] and (rows >= 0).all() and (rows < 1 << 16).all()
