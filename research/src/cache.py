"""Continuous cache (Grave, Joulin, Usunier 2017) and its unbounded form (Grave, Cisse, Joulin 2017)
as an inference-only baseline for any trained model.

At a scored position t the model's final hidden state h_t (after the last norm, before the LM head)
is the query. Every earlier position i < t is a cache entry with key h_i and value x_{i+1} (the token
that followed it, already seen at time t). The cache distribution is

    p_cache(w | t) = sum_{i < t, x_{i+1} = w} exp(theta h_t . h_i) / sum_{i < t} exp(theta h_t . h_i)

and the prediction is the mixture p = (1 - lam) p_model + lam p_cache.

Two modes:
* k = None: exact over every past position ("unbounded"), as a chunked matmul with an online
  log-sum-exp. Only p_cache(target) is needed for the loss, so nothing of vocabulary size is built.
* k = int: the unbounded-cache paper's approximation, softmax over the k nearest past keys by dot
  product (exact top-k here, by chunked matmul and a running top-k merge). The top-k set does not
  depend on theta > 0, so several thetas cost one pass.

All functions take float tensors on any device; nothing here needs CUDA.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _chunks(n, size):
    for s in range(0, n, size):
        yield s, min(n, s + size)


def _scores(hq, qpos, keys, ks, ke):
    """[Qc, Kc] dot products with non-causal entries (key i >= query t) set to -inf."""
    s = hq @ keys[ks:ke].T
    kidx = torch.arange(ks, ke, device=keys.device)
    return s.masked_fill(kidx[None, :] >= qpos[:, None], float("-inf"))


def _topk(hq, qpos, keys, labels, k, k_chunk):
    """Running exact top-k over past keys: (scores [Qc, k], labels [Qc, k]); -inf / -1 pad."""
    Qc = hq.shape[0]
    vals = torch.full((Qc, k), float("-inf"), device=hq.device, dtype=hq.dtype)
    labs = torch.full((Qc, k), -1, device=hq.device, dtype=torch.long)
    tmax = int(qpos.max())
    for ks, ke in _chunks(min(tmax, keys.shape[0]), k_chunk):
        s = _scores(hq, qpos, keys, ks, ke)
        cat = torch.cat([vals, s], 1)
        vals, idx = cat.topk(k, dim=1)
        from_old = idx < k
        old = labs.gather(1, idx.clamp(max=k - 1))
        new = labels[ks:ke][(idx - k).clamp(min=0)]
        labs = torch.where(from_old, old, new)
    labs = labs.masked_fill(torch.isinf(vals), -1)
    return vals, labs


def cache_logp_target(hq, qpos, keys, labels, targets, thetas, k=None, q_chunk=1024, k_chunk=32768):
    """log p_cache(target) at each query, for each theta.

    hq [nq, d] queries (hidden at query positions), qpos [nq] their positions, keys [Lk, d] hidden at
    positions 0..Lk-1, labels [Lk] the token after each key position, targets [nq].
    Key i is visible to a query at position t iff i < t. Returns (lp [n_theta, nq] float32 with
    -inf where no visible key has the target as its label or no key is visible, has [nq] bool: at
    least one visible key).
    """
    nq, nT = hq.shape[0], len(thetas)
    lp = torch.full((nT, nq), float("-inf"), device=hq.device)
    has = torch.zeros(nq, dtype=torch.bool, device=hq.device)
    th = torch.as_tensor(thetas, dtype=hq.dtype, device=hq.device)[:, None, None]
    for qs, qe in _chunks(nq, q_chunk):
        q, qp, tg = hq[qs:qe], qpos[qs:qe], targets[qs:qe]
        has[qs:qe] = qp > 0
        if k is not None:
            vals, labs = _topk(q, qp, keys, labels, k, k_chunk)
            a = th * vals[None]  # [nT, Qc, k]
            logz = torch.logsumexp(a, 2)
            logn = torch.logsumexp(a.masked_fill((labs != tg[:, None])[None], float("-inf")), 2)
            lp[:, qs:qe] = torch.where(torch.isinf(logz), float("-inf"), logn - logz)
            continue
        Qc = q.shape[0]
        m = torch.full((nT, Qc), float("-inf"), device=hq.device)
        Z = torch.zeros(nT, Qc, device=hq.device)
        N = torch.zeros(nT, Qc, device=hq.device)
        for ks, ke in _chunks(min(int(qp.max()), keys.shape[0]), k_chunk):
            s = _scores(q, qp, keys, ks, ke)  # [Qc, Kc]
            match = labels[ks:ke][None, :] == tg[:, None]
            a = th * s[None]  # [nT, Qc, Kc]; theta > 0 keeps -inf as -inf
            mn = torch.maximum(m, a.amax(2))
            mn_safe = torch.where(torch.isinf(mn), torch.zeros_like(mn), mn)
            scale = torch.exp(m - mn_safe)
            e = torch.exp(a - mn_safe[..., None])
            Z = Z * scale + e.sum(2)
            N = N * scale + (e * match[None]).sum(2)
            m = mn
        lp[:, qs:qe] = torch.where(Z > 0, torch.log(N) - torch.log(Z), float("-inf"))
    return lp, has


def cache_logp_vocab(hq, qpos, keys, labels, theta, vocab, k=None, k_chunk=32768):
    """Full log p_cache over the vocabulary [nq, vocab] at one theta (for argmax metrics on a few
    queries, e.g. passkey). Rows with no visible key are -inf everywhere."""
    nq = hq.shape[0]
    if k is not None:
        vals, labs = _topk(hq, qpos, keys, labels, k, k_chunk)
        w = torch.softmax(theta * vals, 1).nan_to_num(0.0)
        acc = torch.zeros(nq, vocab, device=hq.device).scatter_add_(1, labs.clamp(min=0), w * (labs >= 0))
        return torch.log(acc)
    m = torch.full((nq,), float("-inf"), device=hq.device)
    acc = torch.zeros(nq, vocab, device=hq.device)
    for ks, ke in _chunks(min(int(qpos.max()), keys.shape[0]), k_chunk):
        a = theta * _scores(hq, qpos, keys, ks, ke)
        mn = torch.maximum(m, a.amax(1))
        mn_safe = torch.where(torch.isinf(mn), torch.zeros_like(mn), mn)
        acc = acc * torch.exp(m - mn_safe)[:, None]
        acc.scatter_add_(1, labels[ks:ke][None].expand(nq, -1), torch.exp(a - mn_safe[:, None]))
        m = mn
    return torch.log(acc) - torch.log(acc.sum(1, keepdim=True))


def mix_nll(lp_model: np.ndarray, lp_cache: np.ndarray, has: np.ndarray, lam: float) -> np.ndarray:
    """-log((1 - lam) p_model + lam p_cache) per position; positions without a cache use p_model.
    lam = 0 returns -lp_model exactly."""
    if lam == 0:
        return -lp_model
    with np.errstate(divide="ignore"):
        mixed = np.logaddexp(math.log1p(-lam) + lp_model, math.log(lam) + lp_cache)
    return -np.where(has, mixed, lp_model)


def grid_losses(lp_model, lp_cache, has, thetas, lams) -> dict:
    """Mean mixed NLL for every (theta, lam): {str(theta): {str(lam): loss}}."""
    return {f"{t:.6g}": {f"{lam:g}": float(mix_nll(lp_model, lp_cache[j], has, lam).mean()) for lam in lams}
            for j, t in enumerate(thetas)}
