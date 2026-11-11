"""Token tagger (src/tagger.py): each detector on a hand-built sequence with a known answer."""
import numpy as np

from src.tagger import D_FAR, tag_window

# toy vocab: ids 0-9 words, 10 = newline, 11 '(' 12 ')' 13 ',' 14.. integers "1".."9" -> ids 14..22
STRS = [f"w{i}" for i in range(10)] + ["\n", "(", ")", ","] + [str(n) for n in range(1, 10)]
NL, OP, CL, CM = 10, 11, 12, 13
INT = lambda n: 13 + n  # noqa: E731


def _filler(n, rng, lo=0, hi=10):
    return list(rng.integers(lo, hi, size=n))


def test_exact_far_and_near():
    rng = np.random.default_rng(0)
    phrase = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]
    filler = [x + 100 for x in range(D_FAR + 50)]  # unique tokens: no accidental matches
    strs = STRS + [f"u{i}" for i in range(2000)]
    tok = phrase + filler + phrase + phrase
    t = tag_window(tok, strs)
    n = len(phrase)
    far = n + len(filler) + 9  # inside the second copy, match length >= 8
    near = n + len(filler) + n + 9  # inside the third copy: previous copy is n tokens back
    assert t["exact_far"][far] and not t["exact_near"][far]
    assert t["exact_near"][near] and not t["exact_far"][near]


def test_latest_older_consistent():
    # cue (1, 2) followed by 7, then by 8; later (1, 2) -> 8 is latest, (1, 2) -> 7 is older
    # distinct tokens before each (1, 2) so that (1, 2) is the longest earlier match
    seq = [1, 2, 7, 5, 1, 2, 8, 6, 9, 1, 2, 8, 4, 0, 1, 2, 7]
    t = tag_window(seq, STRS)
    assert t["latest_value"][11]   # (1,2) -> 8: conts [7, 8], latest 8
    assert t["older_value"][16]    # (1,2) -> 7: conts [7, 8, 8], latest 8
    seq2 = [3, 4, 9, 0, 3, 4, 9]
    t2 = tag_window(seq2, STRS)
    assert t2["consistent"][6] and not t2["latest_value"][6]


def test_novel_and_no_match():
    t = tag_window([1, 2, 3, 1], STRS)
    assert t["novel"][:3].all() and not t["novel"][3]
    assert t["no_match"][1] and not t["no_match"][0]


def test_bracket_far_and_count_next():
    tok = [OP] + [1] * 70 + [CL]
    assert tag_window(tok, STRS)["bracket_far"][-1]
    assert not tag_window([OP, 1, CL], STRS)["bracket_far"][-1]
    c = tag_window([INT(3), 1, 2, INT(4), 5, INT(7)], STRS)["count_next"]
    assert c[3] and not c[5]


def test_format_template():
    line = lambda a, b, c: [a, CM, b, CM, c, CM, a, NL]  # noqa: E731 - shape w , w , w , w \n
    tok = line(1, 2, 3) + line(4, 5, 6) + line(7, 8, 9)
    f = tag_window(tok, STRS)["format"]
    assert f[8 + 5] and f[16 + 5]  # the comma after "w , w , w" in later lines matches the template


def test_fuzzy_far():
    base = [1, 2, 3, 4, 5, 6, NL]
    edited = [1, 2, 3, 9, 5, 6, NL]  # one token changed: no exact 8-match, high bigram overlap
    strs = STRS + [f"u{i}" for i in range(2000)]
    filler = []
    for i in range(D_FAR // 4 + 20):
        filler += [100 + 3 * i, 101 + 3 * i, 102 + 3 * i, NL]
    tok = base + filler + edited
    fz = tag_window(tok, strs)["fuzzy_far"]
    start = len(base) + len(filler)
    assert fz[start + 4]  # token 5 in the edited line appears in the far near-duplicate line
