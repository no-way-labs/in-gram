"""Real-text language modelling task (WikiText-103, GPT-2 BPE) for the in-context-prior test.

Token ids are shifted by 2 (0 = PAD, 1 = BOS). A batch is BOS + a contiguous window of the token
stream. Every position is scored (next-token prediction). pos_type marks, for each scored
position t, whether the N-gram ending at t (the context locker reads) occurred earlier in the
window: that is the "in-context prior available" subset where locker should help.

Optional Engram-style canonicalization (NFKC + lowercase + strip) maps token ids to canonical
ids used ONLY for hashing (`hash_tokens`); the model still sees raw tokens. Token-level only:
' Paris' and 'Paris' merge, but ' paris' is two GPT-2 tokens and won't.
"""
from __future__ import annotations

import io
import os
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tasks import BOS, IGNORE, PAD

SHIFT = 2
# pos_type codes for scored text positions. A "repeat" is a position whose preceding N-gram (the
# context locker reads) occurred earlier in the window; split by distance to its last occurrence.
PT_TEXT_NOVEL, PT_TEXT_REP_NEAR, PT_TEXT_REP_MID, PT_TEXT_REP_FAR = 10, 11, 12, 13
PT_TEXT_REP = PT_TEXT_REP_NEAR  # back-compat alias (lm-r1..r3 logged all repeats as code 11)
REP_BINS = (256, 1024)  # near < 256 <= mid < 1024 <= far
HF_PARQUET = "https://huggingface.co/api/datasets/Salesforce/wikitext/parquet/{cfg}/{split}/{i}.parquet"
SPLIT_FILES = {"wikitext-103-raw-v1": {"train": 2, "validation": 1, "test": 1},
               "wikitext-2-raw-v1": {"train": 1, "validation": 1, "test": 1}}
CACHE = Path(os.environ.get("TEXT_CACHE", "/tmp/locker_text_cache"))


def _encoder():
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def vocab_size() -> int:
    return _encoder().n_vocab + SHIFT


def load_tokens(cfg: str = "wikitext-103-raw-v1", split: str = "train") -> np.ndarray:
    """Token stream for a split (uint16, shifted), cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{cfg}-{split}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    import pyarrow.parquet as pq

    texts = []
    for i in range(SPLIT_FILES[cfg][split]):
        with urllib.request.urlopen(HF_PARQUET.format(cfg=cfg, split=split, i=i)) as r:
            texts.extend(pq.read_table(io.BytesIO(r.read())).column("text").to_pylist())
    enc = _encoder()
    ids = enc.encode_ordinary_batch(texts, num_threads=min(16, os.cpu_count() or 1))
    arr = np.fromiter((t + SHIFT for doc in ids for t in doc), dtype=np.uint16)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)
    return np.load(path, mmap_mode="r")


PG19_TEST = "https://huggingface.co/api/datasets/emozilla/pg19-test/parquet/default/test/0.parquet"
PG19_VAL = "https://huggingface.co/api/datasets/emozilla/pg19/parquet/default/validation/0.parquet"


def wikitext_style(text: str) -> str:
    """Rewrite plain book text in WikiText-103's raw format, so a WikiText-trained model sees the
    format it knows: hard-wrapped lines joined, paragraphs as ' \n ', punctuation split off by
    spaces, intra-word hyphens as ' @-@ ', possessive ' 's', italics underscores dropped."""
    import re

    text = text.replace("\r", "").replace("_", "")
    text = re.sub(r"\n[ \t]*\n\s*", "\x00", text)  # paragraph breaks
    text = text.replace("\n", " ")  # hard wraps
    for _ in range(2):  # chained hyphens (twenty-five-year)
        text = re.sub(r"(\w)-(\w)", r"\1 @-@ \2", text)
    text = re.sub(r"(\w)'s\b", r"\1 's", text)
    text = re.sub(r'([,.;:!?()"])', r" \1 ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return " " + " \n ".join(p.strip() for p in text.split("\x00") if p.strip()) + " \n"


def _load_pg19(url: str, name: str, min_tokens: int) -> list[np.ndarray]:
    """PG-19 books from one parquet file, WikiText-style, as shifted GPT-2 token arrays, longest first."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npz"
    if not path.exists():
        import pyarrow.parquet as pq

        with urllib.request.urlopen(url) as r:
            texts = pq.read_table(io.BytesIO(r.read())).column("text").to_pylist()
        texts = [wikitext_style(t) for t in texts]
        ids = _encoder().encode_ordinary_batch(texts, num_threads=min(16, os.cpu_count() or 1))
        tmp = CACHE / f"{name}.tmp.npz"
        np.savez(tmp, *[np.asarray(d, dtype=np.uint16) + SHIFT for d in ids])
        os.replace(tmp, path)
    with np.load(path) as z:
        books = [z[k] for k in z.files]
    return sorted((b for b in books if len(b) >= min_tokens), key=len, reverse=True)


def load_pg19_test(min_tokens: int = 0) -> list[np.ndarray]:
    """PG-19 test books (100 public-domain books) as shifted GPT-2 token arrays, longest first."""
    return _load_pg19(PG19_TEST, "pg19-test-wt", min_tokens)


def load_pg19_validation(min_tokens: int = 0) -> list[np.ndarray]:
    """PG-19 validation books (50), same format as load_pg19_test. Used only to tune eval-time
    hyperparameters (the continuous cache's theta and lambda), never for reported test numbers."""
    return _load_pg19(PG19_VAL, "pg19-val-wt", min_tokens)


def canonical_map() -> np.ndarray:
    """canon[id] -> canonical id: tokens whose NFKC-lowercased-stripped text matches share an id."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "gpt2-canon.npy"
    if path.exists():
        return np.load(path)
    enc = _encoder()
    canon = np.arange(enc.n_vocab + SHIFT, dtype=np.int64)
    first = {}
    for t in range(enc.n_vocab):
        try:
            s = enc.decode_single_token_bytes(t).decode("utf-8")
        except UnicodeDecodeError:
            continue  # partial UTF-8 byte tokens stay unique
        key = unicodedata.normalize("NFKC", s).lower().strip()
        if not key:
            key = "<ws>"
        canon[t + SHIFT] = first.setdefault(key, t + SHIFT)
    np.save(path, canon)
    return canon


def repeat_distance(tokens: np.ndarray, n: int) -> np.ndarray:
    """[B, L] int: t - t' where t' < t is the last position whose n-gram (ending there) equals the
    n-gram ending at t; -1 if none (or t < n)."""
    B, L = tokens.shape
    out = np.full((B, L), -1, dtype=np.int64)
    for b in range(B):
        last = {}
        row = tokens[b].tolist()
        for t in range(n, L):
            key = tuple(row[t - n + 1 : t + 1])
            p = last.get(key)
            if p is not None:
                out[b, t] = t - p
            last[key] = t
    return out


def repeat_mask(tokens: np.ndarray, n: int) -> np.ndarray:
    """[B, L] bool: the n-gram ending at t (t >= n, no BOS/PAD) occurred ending at some t' < t."""
    B, L = tokens.shape
    out = np.zeros((B, L), dtype=bool)
    for b in range(B):
        seen = set()
        row = tokens[b]
        for t in range(n, L):
            key = tuple(row[t - n + 1 : t + 1].tolist())
            if key in seen:
                out[b, t] = True
            else:
                seen.add(key)
    return out


@dataclass(frozen=True)
class LMConfig:
    dataset: str = "wikitext-103-raw-v1"
    split: str = "train"
    L_max: int = 1024
    canon: bool = False
    rep_n: int = 2  # N-gram order used to label "in-context prior available" positions


_STREAMS: dict = {}


def lm_batch(cfg: LMConfig, B: int, rng: np.random.Generator) -> dict:
    key = (cfg.dataset, cfg.split)
    if key not in _STREAMS:
        _STREAMS[key] = load_tokens(cfg.dataset, cfg.split)
    stream = _STREAMS[key]
    L = cfg.L_max
    starts = rng.integers(0, len(stream) - L, size=B)
    win = np.stack([np.asarray(stream[s : s + L], dtype=np.int64) for s in starts])  # L tokens
    tokens = np.empty((B, L), dtype=np.int64)
    tokens[:, 0] = BOS
    tokens[:, 1:] = win[:, : L - 1]
    targets = win.copy()  # targets[t] = next token after tokens[t]
    targets[:, 0] = IGNORE  # BOS position: no context yet
    dist = repeat_distance(tokens, cfg.rep_n)
    pos_type = np.full((B, L), PT_TEXT_NOVEL, dtype=np.int8)
    pos_type[(dist > 0) & (dist < REP_BINS[0])] = PT_TEXT_REP_NEAR
    pos_type[(dist >= REP_BINS[0]) & (dist < REP_BINS[1])] = PT_TEXT_REP_MID
    pos_type[dist >= REP_BINS[1]] = PT_TEXT_REP_FAR
    pos_type[:, 0] = 1
    out = dict(tokens=tokens, targets=targets, pos_type=pos_type,
               corrupt=np.zeros((B, L), dtype=bool), src_pos=np.full((B, L), -1, dtype=np.int64))
    if cfg.canon:
        out["hash_tokens"] = canonical_map()[tokens]
    return out
