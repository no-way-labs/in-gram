"""Long-context evaluation of a trained LM checkpoint (inference only, batch 1).

Two measurements per context length L, both on held-out text:

* **Copy at distance** ("probe"): a validation passage P (probe_len tokens) is placed right after
  BOS and again at the very end of an L-token sequence, with training-split filler in between.
  Loss on the second copy vs the first copy (the same targets, no earlier copy) is the copy gain at
  distance ~L. Positions in P whose bigram also occurs in the filler are where cues get crowded.
* **Natural stream**: a window of the held-out stream (validation + test, ~530k tokens), loss by
  bigram repeat distance: novel, < 4k (inside the training window), 4k-32k, >= 32k.

Reports peak GPU memory and wall time per forward. A forward that runs out of memory is recorded
as such and the remaining (longer) lengths for that measurement are skipped.

Every measurement can also run with the continuous cache (src/cache.py; Grave et al. 2017) mixed into
the model's prediction: long_eval(..., cache=cfg) with cfg from tune_cache, or the spec variant
{"cache": {"cache": {...}}} in long_eval_checkpoint (tunes theta and lambda on PG-19 validation books,
then evaluates the test sets with them fixed).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .schedule import build_schedule
from .tasks import BOS

DIST_BINS = ((0, 4096, "rep_lt4k"), (4096, 32768, "rep_4k_32k"), (32768, 1 << 62, "rep_ge32k"))


def _sched_to_torch(sched, device):
    from .train import sched_to_torch

    return sched_to_torch(sched, device)


@torch.no_grad()
def _nll_at(model, scfg, tokens: np.ndarray, pos: np.ndarray, device, cands=None, cache=None,
            top1_cache=False) -> tuple[np.ndarray, dict]:
    """NLL of tokens[p + 1] given tokens[: p + 1] for each p in pos (one sequence, [L] int64).
    info also holds top1 (target is the argmax), rank (1 + number of vocab entries with a strictly
    higher logit) and, if cands (token ids) is given, cand_top1 (target beats every candidate).
    rank and cand_top1 are always the model's own.

    The returned NLL is always the model's own. With cache (dict: thetas, k, and optionally theta_idx
    and lam), info["cache"] = (lp_cache [n_theta, n_pos], has [n_pos]) for the continuous cache, and
    with top1_cache info["top1"] is the argmax of the mixture at (thetas[theta_idx], lam)."""
    from .train import CE_CHUNK

    toks = tokens[None]
    sched = build_schedule(toks, **scfg) if scfg.get("orders") else None  # read_row too if table_rows
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tt = torch.as_tensor(toks, device=device)
    st = _sched_to_torch(sched, device) if sched is not None else None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        h, _ = model(tt, st, return_hidden=True)
    pos_t = torch.as_tensor(pos, device=device)
    hs = h[0, pos_t]
    tg = torch.as_tensor(tokens[pos + 1], device=device)
    ct = torch.as_tensor(np.asarray(cands, dtype=np.int64), device=device) if cands is not None else None
    nll, top1, rank, ctop = [], [], [], []
    for i in range(0, len(pos), CE_CHUNK):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model.head(hs[i:i + CE_CHUNK])
        t = tg[i:i + CE_CHUNK]
        nll.append(F.cross_entropy(lg.float(), t, reduction="none"))
        top1.append(lg.argmax(-1) == t)
        lt = lg.float().gather(1, t[:, None])
        rank.append((lg.float() > lt).sum(-1) + 1)
        if ct is not None:
            ctop.append(lg.float()[:, ct].max(-1).values <= lt[:, 0])
    out = torch.cat(nll).cpu().numpy()
    extra = {}
    if cache:
        from .cache import cache_logp_target, cache_logp_vocab

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        keys, labels, hq = h[0, :-1].float(), tt[0, 1:], hs.float()  # key i -> value tokens[i + 1]
        lpc, has = cache_logp_target(hq, pos_t, keys, labels, tg, cache["thetas"], k=cache.get("k"))
        extra["cache"] = (lpc.cpu().numpy(), has.cpu().numpy())
        if top1_cache and cache.get("theta_idx") is not None and cache["lam"] > 0:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lg = model.head(hs)
            lpm = torch.log_softmax(lg.float(), -1)
            lpv = cache_logp_vocab(hq, pos_t, keys, labels, cache["thetas"][cache["theta_idx"]], lpm.shape[-1],
                                   k=cache.get("k"))
            lam = cache["lam"]
            mixed = torch.logaddexp(lpm + float(np.log1p(-lam)), lpv + float(np.log(lam)))
            mixed = torch.where(has[:, None], mixed, lpm)
            top1 = [mixed.argmax(-1) == tg]
        if device.type == "cuda":
            torch.cuda.synchronize()
        extra["cache_sec"] = round(time.time() - t1, 2)
        del keys, hq
    if device.type == "cuda":
        torch.cuda.synchronize()
    info = dict(sec=round(time.time() - t0, 2),
                peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.type == "cuda" else None,
                n_slots=int(sched["n_slots"].sum()) if sched is not None else 0,
                top1=torch.cat(top1).cpu().numpy(), rank=torch.cat(rank).cpu().numpy(), **extra)
    if ct is not None:
        info["cand_top1"] = torch.cat(ctop).cpu().numpy()
    del h, hs
    return out, info


def _cached(nll, info, cache):
    """(NLL used for the reported metrics, (lp_model, lp_cache, has) for the grid). Without a cache,
    the model NLL; with a tuned cache, the mixture at (thetas[theta_idx], lam)."""
    if not cache:
        return nll, None
    from .cache import mix_nll

    lpc, has = info["cache"]
    part = (-nll, lpc, has)
    if cache.get("theta_idx") is None:
        return nll, part
    return mix_nll(-nll, lpc[cache["theta_idx"]], has, cache["lam"]), part


def _grid(parts, cache):
    """Loss for every (theta, lam) in the cache grid over the concatenated parts."""
    from .cache import grid_losses

    lpm = np.concatenate([p[0] for p in parts])
    lpc = np.concatenate([p[1] for p in parts], 1)
    has = np.concatenate([p[2] for p in parts])
    return grid_losses(lpm, lpc, has, cache["thetas"], cache["lams"])


def _cache_info(infos):
    return dict(cache_sec=max(i.get("cache_sec", 0) for i in infos))


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def probe_eval(model, scfg, device, L, n, probe_len, filler, held, rng, cache=None) -> dict:
    first, second, crowded, infos, p1, p2 = [], [], [], [], [], []
    for _ in range(n):
        p0 = int(rng.integers(0, len(held) - probe_len))
        P = np.asarray(held[p0:p0 + probe_len], dtype=np.int64)
        n_fill = L - 1 - 2 * probe_len
        f0 = int(rng.integers(0, len(filler) - n_fill))
        F_ = np.asarray(filler[f0:f0 + n_fill], dtype=np.int64)
        tokens = np.concatenate([[BOS], P, F_, P])
        s2 = L - probe_len
        pos1 = np.arange(1, probe_len)  # tokens[t] = P[t-1], target P[t]
        pos2 = np.arange(s2, L - 1)  # tokens[pos] = P[i], target P[i+1]
        nll, info = _nll_at(model, scfg, tokens, np.concatenate([pos1, pos2]), device, cache=cache)
        nll, part = _cached(nll, info, cache)
        if part is not None:
            n1 = len(pos1)
            p1.append((part[0][:n1], part[1][:, :n1], part[2][:n1]))
            p2.append((part[0][n1:], part[1][:, n1:], part[2][n1:]))
        first.append(nll[: len(pos1)])
        second.append(nll[len(pos1):])
        # Crowded: the bigram ending at the read position also occurs in the filler.
        # Read position s2 + i holds P[i]; its bigram is (prev, P[i]), prev = last filler token at i = 0.
        fb = set(zip(F_[:-1].tolist(), F_[1:].tolist()))
        prev = np.concatenate([F_[-1:], P[:-1]])
        crowded.append(np.array([(int(prev[i]), int(P[i])) in fb for i in range(probe_len - 1)]))
        infos.append(info)
    a, b, c = np.concatenate(first), np.concatenate(second), np.concatenate(crowded)
    r = dict(L=L, n=n, loss_first=float(a.mean()), loss_second=float(b.mean()), gain=float(a.mean() - b.mean()),
             gain_crowded=float((a[c] - b[c]).mean()) if c.any() else None,
             gain_uncrowded=float((a[~c] - b[~c]).mean()) if (~c).any() else None,
             frac_crowded=float(c.mean()), sec=max(i["sec"] for i in infos),
             peak_gb=max((i["peak_gb"] or 0) for i in infos), n_slots=infos[-1]["n_slots"])
    if cache:  # the whole (theta, lam) grid on this test set: a sensitivity check, never used to select
        g1, g2 = _grid(p1, cache), _grid(p2, cache)
        r.update(_cache_info(infos), cache_grid_second=g2,
                 cache_grid_gain={t: {lam: g1[t][lam] - v for lam, v in row.items()} for t, row in g2.items()})
    return r


def _encode(text: str) -> np.ndarray:
    from .text import SHIFT, _encoder

    return np.asarray(_encoder().encode_ordinary(text), dtype=np.int64) + SHIFT


def passkey_eval(model, scfg, device, L, n, filler, rng, cache=None) -> dict:
    """Passkey retrieval (Mohtashami & Jaggi 2023), in WikiText's format: the needle
    "The pass key is NNNNN . Remember it . NNNNN is the pass key ." follows BOS; training-split
    filler fills the window; the query " The pass key is" ends it. Scored teacher-forced on the
    answer tokens: exact = every answer token is the argmax."""
    exact, nll_ans, infos = [], [], []
    for _ in range(n):
        key = int(rng.integers(10000, 100000))
        needle = _encode(f" The pass key is {key} . Remember it . {key} is the pass key . \n")
        query = _encode(" The pass key is")
        answer = _encode(f" {key}")
        n_fill = L - 1 - len(needle) - len(query) - len(answer)
        f0 = int(rng.integers(0, len(filler) - n_fill))
        tokens = np.concatenate([[BOS], needle, np.asarray(filler[f0:f0 + n_fill], dtype=np.int64), query, answer])
        a0 = len(tokens) - len(answer)
        pos = np.arange(a0 - 1, len(tokens) - 1)  # predict each answer token
        nll, info = _nll_at(model, scfg, tokens, pos, device, cache=cache, top1_cache=True)
        nll, _ = _cached(nll, info, cache)
        exact.append(bool(info["top1"].all()))
        nll_ans.append(float(nll.sum()))
        infos.append(info)
    return dict(L=L, n=n, acc=float(np.mean(exact)), answer_nll=float(np.mean(nll_ans)),
                sec=max(i["sec"] for i in infos), peak_gb=max((i["peak_gb"] or 0) for i in infos),
                **(_cache_info(infos) if cache else {}))


# Word-level retrieval: keys and values are common English words, each a single GPT-2 token with a
# leading space, every one >= 2000 times in WikiText-103 train (so the models can emit them).
WORD_POOL = (
    "red blue green black white gold silver river garden window table chair horse dog cat bird fish king "
    "queen ship train road bridge tower castle forest mountain island ocean lake stone glass paper letter "
    "music piano guitar summer winter spring morning evening night star sun cloud storm snow rain fire "
    "water tree farm village city market church school doctor teacher soldier captain singer engine wheel "
    "chain door wall floor roof box coat hat ring crown flag drum clock salt bear").split()
SCRUB_WORD = "thing"  # replaces needle words where they occur in the filler (not in WORD_POOL)


def _word_ids() -> np.ndarray:
    ids = [_encode(" " + w) for w in WORD_POOL]
    assert all(len(i) == 1 for i in ids), "every pool word must be one token"
    return np.array([int(i[0]) for i in ids], dtype=np.int64)


def _scrubbed_filler(filler, n_fill, rng, words: np.ndarray) -> tuple[np.ndarray, int]:
    """A random n_fill-token training-split window with every token in `words` replaced by
    SCRUB_WORD, so needle words occur only in the needles and the query."""
    if n_fill < 0:
        raise ValueError("needles and query do not fit in L")
    f0 = int(rng.integers(0, len(filler) - n_fill))
    F_ = np.asarray(filler[f0:f0 + n_fill], dtype=np.int64).copy()
    hit = np.isin(F_, words)
    F_[hit] = int(_encode(" " + SCRUB_WORD)[0])
    return F_, int(hit.sum())


def _retrieval_summary(trials: list[dict], infos: list[dict]) -> dict:
    """Pool per-trial answer scores: exact = every answer token is the argmax; rank is 1-based
    (1 = argmax) over the full vocabulary; nll_tok is per answer token."""
    rank = np.concatenate([t["rank"] for t in trials]).astype(np.float64)
    nll = np.concatenate([t["nll"] for t in trials])
    return dict(acc=float(np.mean([t["exact"] for t in trials])), rank_mean=float(rank.mean()),
                rank_median=float(np.median(rank)), log_rank_mean=float(np.log(rank).mean()),
                top10=float((rank <= 10).mean()), nll_tok=float(nll.mean()),
                answer_nll=float(np.mean([t["nll"].sum() for t in trials])),
                sec=max(i["sec"] for i in infos), peak_gb=max((i["peak_gb"] or 0) for i in infos),
                scrubbed=float(np.mean([t["scrubbed"] for t in trials])))


def wordkey_eval(model, scfg, device, L, n, filler, rng, n_words: int = 3) -> dict:
    """passkey_eval with a word key: the needle "The pass key is red river clock . Remember it .
    red river clock is the pass key ." follows BOS, filler (pool words scrubbed) fills the window,
    the query " The pass key is" ends it; the answer is the n_words key words (distinct pool words).
    Per answer position: top-1 rate, mean rank, NLL (position 0 is retrieval; later positions can
    also be copied from the needle given the first word)."""
    ids = _word_ids()
    trials, infos = [], []
    for _ in range(n):
        key = " ".join(WORD_POOL[i] for i in rng.choice(len(WORD_POOL), n_words, replace=False))
        needle = _encode(f" The pass key is {key} . Remember it . {key} is the pass key . \n")
        query = _encode(" The pass key is")
        answer = _encode(f" {key}")
        assert len(answer) == n_words
        F_, scrubbed = _scrubbed_filler(filler, L - 1 - len(needle) - len(query) - len(answer), rng, ids)
        tokens = np.concatenate([[BOS], needle, F_, query, answer])
        a0 = len(tokens) - len(answer)
        nll, info = _nll_at(model, scfg, tokens, np.arange(a0 - 1, len(tokens) - 1), device)
        trials.append(dict(exact=bool(info["top1"].all()), top1=info["top1"], rank=info["rank"], nll=nll,
                           scrubbed=scrubbed))
        infos.append(info)
    by = lambda k: np.stack([t[k] for t in trials]).astype(np.float64).mean(0).tolist()  # noqa: E731
    return dict(L=L, n=n, words=n_words, **_retrieval_summary(trials, infos),
                top1_by_pos=by("top1"), rank_by_pos=by("rank"), nll_by_pos=by("nll"))


def multikey_eval(model, scfg, device, L, n, filler, rng, pairs: int = 8) -> dict:
    """RULER-style multi-key retrieval: `pairs` needles " The special word for KEY is VALUE ." at
    random depths in the filler (each moved to just after the next " ." within 256 tokens, so it
    starts a sentence), then the query " The special word for KEY is" for one of the keys, answer
    " VALUE". Keys and values are 2 * pairs distinct pool words, scrubbed from the filler, so each
    key occurs only in its needle and the query. acc_choice: the right value beats the other
    in-context values (chance 1 / pairs). depth: queried needle's start / L."""
    ids = _word_ids()
    dot = int(_encode(" .")[0])
    trials, infos = [], []
    for _ in range(n):
        w = rng.choice(len(WORD_POOL), 2 * pairs, replace=False)
        keys, vals = [WORD_POOL[i] for i in w[:pairs]], [WORD_POOL[i] for i in w[pairs:]]
        needles = [_encode(f" The special word for {k} is {v} .") for k, v in zip(keys, vals)]
        q = int(rng.integers(pairs))
        query = _encode(f" The special word for {keys[q]} is")
        answer = _encode(f" {vals[q]}")
        n_fill = L - 1 - sum(len(x) for x in needles) - len(query) - len(answer)
        F_, scrubbed = _scrubbed_filler(filler, n_fill, rng, ids)
        cut = np.sort(rng.integers(0, n_fill + 1, size=pairs))
        dots = np.flatnonzero(F_ == dot)
        j = np.searchsorted(dots, cut)
        for i in range(pairs):  # snap to a sentence start when one is near
            if j[i] < len(dots) and dots[j[i]] - cut[i] < 256:
                cut[i] = dots[j[i]] + 1
        cut = np.sort(cut)
        order = rng.permutation(pairs)  # needle order independent of depth sort
        parts, starts, prev, at = [[BOS]], {}, 0, 1
        for i in range(pairs):
            parts.append(F_[prev:cut[i]])
            at += cut[i] - prev
            starts[int(order[i])] = at
            parts.append(needles[order[i]])
            at += len(needles[order[i]])
            prev = cut[i]
        parts += [F_[prev:], query, answer]
        tokens = np.concatenate(parts).astype(np.int64)
        assert len(tokens) == L
        a0 = len(tokens) - len(answer)
        cands = np.array([int(_encode(" " + v)[0]) for v in vals])
        nll, info = _nll_at(model, scfg, tokens, np.arange(a0 - 1, len(tokens) - 1), device, cands=cands)
        trials.append(dict(exact=bool(info["top1"].all()), rank=info["rank"], nll=nll, scrubbed=scrubbed,
                           choice=bool(info["cand_top1"].all()), depth=starts[q] / L))
        infos.append(info)
    return dict(L=L, n=n, pairs=pairs, **_retrieval_summary(trials, infos),
                acc_choice=float(np.mean([t["choice"] for t in trials])), chance_choice=1.0 / pairs,
                per_trial=[dict(depth=round(t["depth"], 4), exact=t["exact"], choice=t["choice"],
                                rank=int(t["rank"][0]), nll=round(float(t["nll"][0]), 4)) for t in trials])


def stream_eval(model, scfg, device, L, held, n, cache=None) -> dict:
    from .text import repeat_distance

    sums, counts, infos, parts = {}, {}, [], []
    for w in range(n):
        s0 = w * (L - 1)
        if s0 + L - 1 > len(held):
            break
        win = np.asarray(held[s0:s0 + L - 1], dtype=np.int64)
        tokens = np.concatenate([[BOS], win])
        pos = np.arange(1, L - 1)
        nll, info = _nll_at(model, scfg, tokens, pos, device, cache=cache)
        nll, part = _cached(nll, info, cache)
        if part is not None:
            parts.append(part)
        dist = repeat_distance(tokens[None], 2)[0][pos]
        labels = np.full(len(pos), "novel", dtype=object)
        for lo, hi, name in DIST_BINS:
            labels[(dist >= max(lo, 1)) & (dist < hi)] = name
        for name in ["novel"] + [b[2] for b in DIST_BINS]:
            m = labels == name
            sums[name] = sums.get(name, 0.0) + float(nll[m].sum())
            counts[name] = counts.get(name, 0) + int(m.sum())
        infos.append(info)
    if not infos:
        return dict(L=L, skipped="held-out stream too short")
    tot = sum(counts.values())
    r = dict(L=L, n=len(infos), loss=sum(sums.values()) / tot,
             loss_by_bin={k: sums[k] / counts[k] for k in sums if counts[k]},
             frac_by_bin={k: counts[k] / tot for k in counts},
             sec=max(i["sec"] for i in infos), peak_gb=max((i["peak_gb"] or 0) for i in infos))
    if cache:
        r.update(_cache_info(infos), cache_grid=_grid(parts, cache))
    return r


def _bin_losses(nll, tokens, pos):
    from .text import repeat_distance

    dist = repeat_distance(tokens[None], 2)[0][pos]
    labels = np.full(len(pos), "novel", dtype=object)
    for lo, hi, name in DIST_BINS:
        labels[(dist >= max(lo, 1)) & (dist < hi)] = name
    return {name: (float(nll[labels == name].sum()), int((labels == name).sum()))
            for name in ["novel"] + [b[2] for b in DIST_BINS]}


def doc_eval(model, scfg, device, books, max_len, cache=None) -> dict:
    """Whole books (PG-19 test), each truncated to max_len: loss by bigram repeat distance within
    one long document (repeats come from the same book, unlike the concatenated-article stream)."""
    sums, counts, per_book, infos, parts = {}, {}, [], [], []
    for book in books:
        win = np.asarray(book[: max_len - 1], dtype=np.int64)
        tokens = np.concatenate([[BOS], win])
        pos = np.arange(1, len(tokens) - 1)
        nll, info = _nll_at(model, scfg, tokens, pos, device, cache=cache)
        nll, part = _cached(nll, info, cache)
        if part is not None:
            parts.append(part)
        for k, (sm, c) in _bin_losses(nll, tokens, pos).items():
            sums[k] = sums.get(k, 0.0) + sm
            counts[k] = counts.get(k, 0) + c
        per_book.append(dict(L=len(tokens), loss=float(nll.mean()), sec=info["sec"], peak_gb=info["peak_gb"]))
        infos.append(info)
    tot = sum(counts.values())
    r = dict(max_len=max_len, n=len(books), loss=sum(sums.values()) / tot,
             loss_by_bin={k: sums[k] / counts[k] for k in sums if counts[k]},
             frac_by_bin={k: counts[k] / tot for k in counts}, per_book=per_book,
             peak_gb=max((i["peak_gb"] or 0) for i in infos))
    if cache:
        r.update(_cache_info(infos), cache_grid=_grid(parts, cache))
    return r


def long_eval(model, scfg, device, spec: dict, seed: int = 4242, cache=None) -> dict:
    """spec: probe_lengths, probe_n (int or {L: n}), probe_len, stream_lengths, stream_n,
    passkey_lengths, passkey_n, wordkey_lengths, wordkey_n, wordkey_words, multikey_lengths,
    multikey_n, multikey_pairs (int or list: one multikey record per pairs value).
    cache: a tuned continuous-cache config (tune_cache(...)["eval"]) mixed into every measurement
    except the word-key and multi-key evals (model only)."""
    from .text import load_tokens

    filler = load_tokens("wikitext-103-raw-v1", "train")
    held = np.concatenate([load_tokens("wikitext-103-raw-v1", "validation"),
                           load_tokens("wikitext-103-raw-v1", "test")]).astype(np.int64)
    model.eval()
    out = dict(probe=[], stream=[], doc=[], passkey=[], wordkey=[], multikey=[])
    books = []
    if spec.get("doc_n"):
        from .text import load_pg19_test

        for attempt in range(3):  # a flaky download must not take the other evals down with it
            try:
                books = load_pg19_test()[: spec["doc_n"]]
                break
            except Exception:
                import traceback

                if attempt == 2:
                    out["doc"].append(dict(error="PG-19 load failed: " + traceback.format_exc()[-600:]))
                time.sleep(10 * (attempt + 1))
    if books:
        for max_len in spec.get("doc_max", [1 << 20]):
            try:
                out["doc"].append(doc_eval(model, scfg, device, books, max_len, cache=cache))
            except Exception as e:
                import traceback

                out["doc"].append(dict(max_len=max_len, oom=_is_oom(e), error=traceback.format_exc()[-1500:]))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for kind, lengths in (("probe", spec.get("probe_lengths", [])), ("stream", spec.get("stream_lengths", [])),
                          ("passkey", spec.get("passkey_lengths", [])),
                          ("wordkey", spec.get("wordkey_lengths", [])),
                          ("multikey", spec.get("multikey_lengths", []))):
        for L in sorted(lengths):
            rng = np.random.default_rng(seed + L)
            try:
                if kind == "probe":
                    pn = spec.get("probe_n", 4)
                    n = pn.get(str(L), pn.get(L, 2)) if isinstance(pn, dict) else pn
                    r = probe_eval(model, scfg, device, L, n, spec.get("probe_len", 512), filler, held, rng, cache=cache)
                elif kind == "passkey":
                    r = passkey_eval(model, scfg, device, L, spec.get("passkey_n", 5), filler, rng, cache=cache)
                elif kind == "wordkey":
                    r = wordkey_eval(model, scfg, device, L, spec.get("wordkey_n", 5), filler, rng,
                                     n_words=spec.get("wordkey_words", 3))
                elif kind == "multikey":
                    mp = spec.get("multikey_pairs", 8)
                    r = [multikey_eval(model, scfg, device, L, spec.get("multikey_n", 5), filler,
                                       np.random.default_rng((seed, L, p)), pairs=p)  # independent of the list
                         for p in (mp if isinstance(mp, (list, tuple)) else [mp])]
                else:
                    r = stream_eval(model, scfg, device, L, held, spec.get("stream_n", 1), cache=cache)
            except Exception as e:  # record the failure; keep every other length's result
                import traceback

                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if _is_oom(e):  # longer lengths will OOM too
                    out[kind].append(dict(L=L, oom=True, error=str(e)[:300]))
                    break
                out[kind].append(dict(L=L, error=traceback.format_exc()[-1500:]))
                continue
            out[kind].extend(r if isinstance(r, list) else [r])
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return out


# Continuous-cache defaults. theta is set relative to the model's hidden scale: theta = c / E||h||^2
# (E over the first 4k tokens of the first tuning book), so c = 1 is a near-uniform cache over the
# past and c = 300 is close to nearest-neighbour lookup, whatever the final norm's learned gain.
CACHE_THETA_REL = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
CACHE_LAMS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)


@torch.no_grad()
def _hidden_sq_norm(model, scfg, tokens, device) -> float:
    sched = build_schedule(tokens[None], **scfg) if scfg.get("orders") else None
    st = _sched_to_torch(sched, device) if sched is not None else None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        h, _ = model(torch.as_tensor(tokens[None], device=device), st, return_hidden=True)
    return float((h[0].float() ** 2).sum(-1).mean())


def tune_cache(model, scfg, device, cfg: dict, books=None) -> dict:
    """Pick the continuous cache's theta and lambda on held-out validation text, as Grave et al. did.

    cfg: theta_rel (grid of c in theta = c / E||h||^2), lams, k (None = exact over every past
    position), tune_n (PG-19 validation books, longest first), tune_max_len (tokens per book),
    test_grid (True: evaluate the whole grid on test sets too, reported as a sensitivity check).
    books: token arrays to tune on instead of PG-19 validation (tests). Returns the tuning record;
    its "eval" entry is the config long_eval(..., cache=) takes."""
    source = "given" if books is not None else "pg19-validation"
    if books is None:
        from .text import load_pg19_validation

        for attempt in range(3):
            try:
                books = load_pg19_validation()[: cfg.get("tune_n", 4)]
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(10 * (attempt + 1))
    model.eval()
    max_len = cfg.get("tune_max_len", 131072)
    rel = list(cfg.get("theta_rel", CACHE_THETA_REL))
    lams = list(cfg.get("lams", CACHE_LAMS))
    first = np.concatenate([[BOS], np.asarray(books[0][: min(4096, max_len) - 1], dtype=np.int64)])
    h2 = _hidden_sq_norm(model, scfg, first, device)
    thetas = [c / h2 for c in rel]
    probe = dict(thetas=thetas, lams=lams, k=cfg.get("k"))
    parts, t0 = [], time.time()
    for book in books:
        tokens = np.concatenate([[BOS], np.asarray(book[: max_len - 1], dtype=np.int64)])
        nll, info = _nll_at(model, scfg, tokens, np.arange(1, len(tokens) - 1), device, cache=probe)
        parts.append(_cached(nll, info, probe)[1])
        if device.type == "cuda":
            torch.cuda.empty_cache()
    grid = _grid(parts, probe)
    flat = [(v, j, lam) for j, t in enumerate(thetas) for lam, v in zip(lams, grid[f"{t:.6g}"].values())]
    best, j, lam = min(flat, key=lambda x: (x[0], x[2]))
    ev = dict(thetas=thetas if cfg.get("test_grid", True) else [thetas[j]],
              lams=lams if cfg.get("test_grid", True) else [lam], k=cfg.get("k"),
              theta_idx=j if cfg.get("test_grid", True) else 0, lam=lam)
    return dict(theta=thetas[j], theta_rel=rel[j], lam=lam, k=cfg.get("k"), h_sq_norm=h2,
                tune_loss=best, tune_loss_nocache=grid[f"{thetas[j]:.6g}"]["0"] if 0 in lams else None,
                tune_data=source,
                tune_n=len(books), tune_max_len=max_len, tune_tokens=int(sum(len(p[0]) for p in parts)),
                tune_sec=round(time.time() - t0, 1), theta_rel_grid=rel, theta_grid=thetas, grid=grid, eval=ev)


def needs_schedule(mcfg: dict) -> bool:
    """Whether a model reads hashed cues (overlay, static table, or hgdn hashed columns)."""
    return bool(mcfg.get("overlay") or mcfg.get("static") or (mcfg.get("hash") or {}).get("orders"))


def long_eval_checkpoint(ckpt_path, spec: dict) -> dict:
    """Rebuild the model from a train.run checkpoint (which stores mcfg/scfg) and run long_eval."""
    from .models import Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = dict(ck["mcfg"])
    model = Model(mcfg).to(device)
    model.load_state_dict(ck["model"])
    scfg = ck["scfg"] if needs_schedule(mcfg) else {}
    out = long_eval(model, scfg, device, spec)
    for name, mod in model.named_modules():  # Bayesian locker: learned confidence per cue row
        if getattr(mod, "prior_rows", None):
            if mod.kappa_mode == "row":
                lk = mod.kappa_table.detach()
            else:  # kappa for every prior row of every stream
                OH, R = mod.prior_table.shape[:2]
                st = torch.arange(OH, device=mod.prior_table.device)[:, None].expand(OH, R)
                rw = torch.arange(R, device=mod.prior_table.device)[None].expand(OH, R)
                lk = mod.log_kappa(st, rw, mod.prior_table.detach())
            k = lk.detach().exp().float().cpu()
            q = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99])
            out.setdefault("kappa", {})[name] = dict(
                quantiles=k.flatten().quantile(q).tolist(),  # over all rows (untouched rows stay at init)
                per_stream_median=k.median(dim=1).values.tolist(), mean=float(k.mean()))
    # Inference-time variants of the same weights, e.g. {"backoff_all": {"backoff": True}} (lockers
    # only) or {"cache": {"cache": CACHE_CFG}} (continuous cache, any model). attrs["spec"] overrides
    # spec keys for that variant.
    lockers = [m for m in model.modules() if hasattr(m, "set_backoff")]
    for vname, attrs in (spec.get("variants") or {}).items():
        vspec = {k: v for k, v in {**spec, **attrs.get("spec", {})}.items() if k != "variants"}
        if "cache" in attrs:
            import traceback

            try:
                tuned = tune_cache(model, scfg, device, attrs["cache"])
                r = long_eval(model, scfg, device, vspec, cache=tuned["eval"])
                r["cache"] = tuned
            except Exception:
                r = dict(error=traceback.format_exc()[-3000:])
            out.setdefault("variants", {})[vname] = r
            continue
        if not lockers:
            continue
        saved = [(m, m.backoff, getattr(m, "_backoff_sel", None)) for m in lockers]
        for m in lockers:
            m.set_backoff(attrs["backoff"])
        out.setdefault("variants", {})[vname] = long_eval(model, scfg, device, vspec)
        for m, b, sel in saved:
            m.backoff = b
            if sel is not None:
                m._backoff_sel = sel
    rope_attn = [m for m in model.modules() if getattr(m, "rope", False) and hasattr(m, "ntk_train_len")]
    if spec.get("ntk_train_len") and rope_attn:  # second pass with dynamic NTK RoPE scaling
        for m in rope_attn:
            m.ntk_train_len = spec["ntk_train_len"]
        out["ntk"] = long_eval(model, scfg, device, spec)
    return out
