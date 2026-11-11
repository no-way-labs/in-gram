"""Hand-constructed sequences with known pointers."""
import numpy as np
import pytest

from src.hashing import ngram_addresses
from src.schedule import build_schedule, set_and_tag
from src.tasks import BOS, PAD


def ptr_pos(sched, b, s, oi=0, h=0):
    p = sched["read_ptr"][b, s, oi, h]
    return None if p < 0 else int(sched["ev_pos"][p])


@pytest.mark.parametrize("mode", ["growing", "bounded"])
def test_kv_pointer(mode):
    # BOS k1 k2 v n n k1 k2 ?   -> read at pos 7 (k2) points to write at pos 3 (v)
    toks = np.array([[BOS, 10, 11, 90, 140, 141, 10, 11, 92, PAD]])
    s = build_schedule(toks, orders=(2,), H=2, mode=mode, M=64, ways=4)
    for h in range(2):
        assert ptr_pos(s, 0, 7, 0, h) == 3
        assert ptr_pos(s, 0, 2, 0, h) is None  # first occurrence: empty
    # no write at PAD position 9, and no event addressed by a window with BOS
    assert 9 not in s["ev_pos"].tolist() and 1 not in s["ev_pos"].tolist() and 2 not in s["ev_pos"].tolist()


@pytest.mark.parametrize("mode", ["growing", "bounded"])
def test_adjacent_repeats(mode):
    # "a a a a": bigram (a,a) ends at 2,3,4. write at p=3 -> slot(aa ending 2), p=4 -> same slot
    toks = np.array([[BOS, 5, 5, 5, 5, 5]])
    s = build_schedule(toks, orders=(2,), H=1, mode=mode, M=64, ways=4)
    assert ptr_pos(s, 0, 2) is None
    assert ptr_pos(s, 0, 3) is None  # write at p=3 is not visible to read at s=3
    assert ptr_pos(s, 0, 4) == 3
    assert ptr_pos(s, 0, 5) == 4
    # order 1: unigram (a) ends at 1..; write at p=2 -> slot(a)
    s1 = build_schedule(toks, orders=(1,), H=1, mode=mode, M=64, ways=4)
    assert ptr_pos(s1, 0, 1) is None and ptr_pos(s1, 0, 2) is None
    assert ptr_pos(s1, 0, 3) == 2 and ptr_pos(s1, 0, 5) == 4


@pytest.mark.parametrize("mode", ["growing", "bounded"])
def test_write_read_rewrite(mode):
    # k: (10,11). write v1 at 3, read at 5, rewrite at 6 (v2), read at 8 -> 6
    toks = np.array([[BOS, 10, 11, 90, 10, 11, 91, 10, 11]])
    s = build_schedule(toks, orders=(2,), H=1, mode=mode, M=64, ways=4)
    assert ptr_pos(s, 0, 5) == 3
    assert ptr_pos(s, 0, 8) == 6
    # same virtual slot: pointers at 5 and 8 lie in one segment
    p5, p8 = s["read_ptr"][0, 5, 0, 0], s["read_ptr"][0, 8, 0, 0]
    assert s["ev_seg_start"][p5] == s["ev_seg_start"][p8]


def _colliding_tokens(n_keys, M, ways):
    """Find distinct bigrams that all map to the same cache set."""
    n_sets = M // ways
    target, found = None, []
    for a in range(2, 66):
        for b in range(2, 66):
            addr, _ = ngram_addresses(np.array([[BOS, a, b]]), 2, 0)
            st = int(set_and_tag(addr[0, 2], M, ways)[0])
            if target is None:
                target = st
            if st == target:
                found.append((a, b))
            if len(found) == n_keys:
                return found
    raise AssertionError


def test_bounded_eviction_creates_new_epoch():
    M, ways = 8, 2
    n_sets = M // ways
    keys = _colliding_tokens(3, M, ways)
    key_bigrams = set(keys)
    target = None
    rng = np.random.default_rng(0)
    # write k0, k1, k2 (evicts k0, the LRU), then read k0 -> empty; read k2 -> hit.
    # Pick value tokens so no other bigram lands in the keys' cache set.
    for _ in range(10_000):
        vals = rng.integers(66, 130, size=3)
        seq = [BOS]
        for i, (a, b) in enumerate(keys):
            seq += [a, b, int(vals[i])]
        seq += [keys[0][0], keys[0][1], keys[2][0], keys[2][1]]
        addr, valid = ngram_addresses(np.array([seq]), 2, 0)
        sets = set_and_tag(addr[0], M, ways)[0]
        target = sets[2]
        others = [t for t in range(2, len(seq)) if tuple(seq[t - 1 : t + 1]) not in key_bigrams]
        if all(sets[t] != target for t in others):
            break
    else:
        raise AssertionError("no clean sequence found")
    s = build_schedule(np.array([seq]), orders=(2,), H=1, mode="bounded", M=M, ways=ways)
    L = len(seq)
    assert ptr_pos(s, 0, L - 3) is None  # k0 evicted
    assert ptr_pos(s, 0, L - 1) == 9  # k2's value position
    g = build_schedule(np.array([seq]), orders=(2,), H=1, mode="growing")
    assert ptr_pos(g, 0, L - 3) == 3  # growing mode never evicts


def test_random_reads_point_earlier_same_stream():
    rng = np.random.default_rng(0)
    toks = rng.integers(2, 200, size=(4, 64))
    toks[:, 0] = BOS
    s = build_schedule(toks, orders=(1, 2), H=2, read="random", seed=3)
    rp = s["read_ptr"]
    for b, t, o, h in zip(*np.nonzero(rp >= 0)):
        e = rp[b, t, o, h]
        assert s["ev_b"][e] == b and s["ev_pos"][e] < t and s["ev_order"][e] == o


def test_tags_and_sets_are_diverse():
    """Regression (DEVIATIONS D4): raw bigram addresses share their high bits."""
    toks = np.array([[BOS] + [a for a in range(2, 18) for _ in range(1)] * 2])
    addr, valid = ngram_addresses(toks, 2, 0)
    a = addr[valid]
    st, tg, _ = set_and_tag(a, 256, 4)
    assert len(set(tg.tolist())) == len(set(a.tolist()))
    assert len(set(st.tolist())) > 0.5 * min(64, len(set(a.tolist())))
