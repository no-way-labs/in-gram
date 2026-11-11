"""Dependency-type tags for target tokens (docs/CAPABILITY_MATRIX.md rows), from tokens alone.

`tag_window(tok, strs)` returns {tag: bool [T]} where position t is the target tok[t] predicted from
tok[:t] (same convention as src/pointer.py). Tags overlap; a position can carry several. v1 covers
the rows detectable without parsing; multi-hop (row 9), position-addressed (row 6) and state
tracking (row 8) need code-aware detectors or synthetic probes (v2).

Tags (D_FAR = 512 tokens):
  exact_far      longest earlier match >= 8 tokens, the most recent occurrence's continuation is the
                 target, and that occurrence is > D_FAR back (row 2, beyond a sliding window)
  exact_near     the same within D_FAR (row 2, inside a window)
  consistent     short match (2-7 tokens): the cue's earlier continuations all equal the target
                 (row 4: a stable association)
  latest_value   short match: the cue has >= 2 distinct earlier continuations and the most recent
                 one is the target (row 5: recency decides)
  older_value    short match: >= 2 distinct earlier continuations, the target is one of them but
                 not the most recent (row 5 inverse: recency misleads; locker5's motivating case)
  novel          the target token never appeared earlier in the window (row 12-13: nothing to copy)
  no_match       no earlier match of even one token of context (pointer regime "none")
  fuzzy_far      no exact match >= 8, but the current line is a near-duplicate (token-bigram Jaccard
                 >= 0.5) of a line > D_FAR back that contains the target (row 3)
  bracket_far    the target closes a bracket whose opener is > 64 tokens back (row 7, scope)
  count_next     the target is an integer n and the nearest earlier integer within 200 tokens is n - 1
                 (row 10, counting / enumeration)
  format         no exact match >= 8; the target is punctuation/whitespace, and an earlier line with
                 the same token-class shape (>= 5 tokens) continued with the same token (row 14)
"""
from __future__ import annotations

import re

import numpy as np

from .pointer import LADDER, match_lengths, suffix_keys

D_FAR = 512
TAGS = ("exact_far", "exact_near", "consistent", "latest_value", "older_value", "novel", "no_match",
        "fuzzy_far", "bracket_far", "count_next", "format")
_INT = re.compile(r"^\s*\d{1,6}\s*$")
_PUNCT = re.compile(r"^[\W_]*$")
_OPEN, _CLOSE = "([{", ")]}"


def token_strings(tokenizer: str = "gpt2"):
    """id -> str table for the agentic tokenizer (ids shifted by SHIFT; ids below SHIFT -> '')."""
    from .agentic import SHIFT, get_tokenizer

    tk = get_tokenizer(tokenizer)
    out = [""] * tk.vocab
    if hasattr(tk, "enc"):
        for i in range(tk.n_vocab):
            out[i + SHIFT] = tk.enc.decode_single_token_bytes(i).decode("utf-8", "replace")
    else:
        for i in range(tk.n_vocab):
            out[i + SHIFT] = tk.tok.id_to_token(i) or ""
    return out


def _cue_tags(tok, match_len):
    """consistent / latest_value / older_value from all earlier continuations of the matched cue."""
    T = len(tok)
    keys = suffix_keys(tok, LADDER)
    hist = {L: {} for L in LADDER}
    cons = np.zeros(T, bool)
    latest = np.zeros(T, bool)
    older = np.zeros(T, bool)
    for t in range(1, T):
        L = int(match_len[t])
        if 2 <= L <= 7:  # short matches only: long matches are the exact_* tags
            conts = hist[L].get(int(keys[L][t]))
            if conts:
                y = tok[t]
                if len(set(conts)) == 1:
                    cons[t] = conts[0] == y
                elif conts[-1] == y:
                    latest[t] = True
                elif y in conts:
                    older[t] = True
        for L2 in LADDER:  # record this position's continuation under every cue length that ends at t-1
            if t >= L2:
                hist[L2].setdefault(int(keys[L2][t]), []).append(tok[t])
    return cons, latest, older


def _lines(strs_t):
    """line id per position (a token containing a newline ends its line), and line start positions."""
    T = len(strs_t)
    line = np.zeros(T, np.int64)
    starts = [0]
    cur = 0
    for t in range(T):
        line[t] = cur
        if "\n" in strs_t[t]:
            cur += 1
            starts.append(t + 1)
    return line, starts


def _fuzzy_far(tok, line, starts, exact_long):
    T = len(tok)
    out = np.zeros(T, bool)
    n_lines = int(line[-1]) + 1 if T else 0
    bounds = list(starts) + [T]
    big = {}  # bigram -> list of line ids containing it
    sets, toks = [], []
    for li in range(n_lines):
        a, b = bounds[li], bounds[li + 1]
        bg = set(zip(tok[a:b - 1].tolist(), tok[a + 1:b].tolist()))
        sets.append(bg)
        toks.append(set(tok[a:b].tolist()))
    for li in range(n_lines):
        a, b = bounds[li], bounds[li + 1]
        bg = sets[li]
        if len(bg) >= 4:
            counts = {}
            for x in bg:
                for lj in big.get(x, ()):
                    counts[lj] = counts.get(lj, 0) + 1
            best = None
            for lj, c in counts.items():
                if bounds[lj + 1] <= a - D_FAR and c / len(bg | sets[lj]) >= 0.5:
                    best = lj
                    break
            if best is not None:
                for t in range(a, b):
                    if not exact_long[t] and tok[t] in toks[best]:
                        out[t] = True
        for x in bg:
            lst = big.setdefault(x, [])
            if len(lst) < 64:
                lst.append(li)
    return out


def _bracket_far(strs_t):
    T = len(strs_t)
    out = np.zeros(T, bool)
    stack = []
    for t, s in enumerate(strs_t):
        for ch in s:
            if ch in _OPEN:
                stack.append((ch, t))
            elif ch in _CLOSE and stack:
                o, pos = stack.pop()
                if _OPEN.index(o) == _CLOSE.index(ch) and t - pos > 64:
                    out[t] = True
    return out


def _count_next(strs_t):
    T = len(strs_t)
    out = np.zeros(T, bool)
    last_int, last_pos = None, -10**9
    for t, s in enumerate(strs_t):
        if _INT.match(s):
            n = int(s)
            if last_int is not None and t - last_pos <= 200 and n == last_int + 1:
                out[t] = True
            last_int, last_pos = n, t
    return out


def _cls(s):
    if _INT.match(s):
        return "#"
    if _PUNCT.match(s):
        return s  # punctuation / whitespace kept verbatim
    return "w"


def _format(tok, strs_t, line, starts):
    T = len(tok)
    out = np.zeros(T, bool)
    seen = {}  # shape-prefix -> continuation token of the most recent line with that prefix
    shape = ()
    cur_line = 0
    for t in range(T):
        if line[t] != cur_line:
            cur_line, shape = int(line[t]), ()
        if len(shape) >= 5 and _PUNCT.match(strs_t[t]) and seen.get(shape) == tok[t]:
            out[t] = True
        if len(shape) >= 5:
            seen[shape] = tok[t]
        shape = shape + (_cls(strs_t[t]),)
        if len(shape) > 12:
            shape = shape[-12:]
    return out


def tag_window(tok, strs) -> dict:
    tok = np.asarray(tok, dtype=np.int64)
    T = len(tok)
    strs_t = [strs[int(i)] if 0 <= int(i) < len(strs) else "" for i in tok]
    match_len, src = match_lengths(tok)
    has = src >= 0
    copy_ok = np.zeros(T, bool)
    copy_ok[has] = tok[src[has]] == tok[has]
    dist = np.where(has, np.arange(T) - src, 0)
    long_ok = (match_len >= 8) & copy_ok
    cons, latest, older = _cue_tags(tok, match_len)
    first = np.ones(T, bool)
    _, idx = np.unique(tok, return_index=True)
    first[:] = False
    first[idx] = True
    line, starts = _lines(strs_t)
    tags = dict(
        exact_far=long_ok & (dist > D_FAR),
        exact_near=long_ok & (dist <= D_FAR),
        consistent=cons,
        latest_value=latest,
        older_value=older,
        novel=first,
        no_match=match_len == 0,
        fuzzy_far=_fuzzy_far(tok, line, starts, match_len >= 8),
        bracket_far=_bracket_far(strs_t),
        count_next=_count_next(strs_t),
        format=_format(tok, strs_t, line, starts) & (match_len < 8),
    )
    tags["no_match"][0] = False
    return tags
