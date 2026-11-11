"""Agentic trajectories (nebius/SWE-rebench-openhands-trajectories) as a language-modelling task.

Pipeline:
  1. `render`: one trajectory (OpenAI-style messages) -> one string, in message order: message text,
     each tool call as JSON of {name, arguments}, tool outputs verbatim. The system prompt (identical OpenHands boilerplate in every row) is dropped by
     default; `system=True` keeps it.
  2. `repo_split`: train / eval by a stable hash of the `repo` field (eval ~5% of repos), so no
     repository appears in both splits (trajectories on one repo share file contents).
  3. `build`: stream the parquet file from the HF API URL (HTTP range reads, one row group at a
     time; no full download needed), render, tokenize with a pluggable tokenizer, and cache one
     token shard per row group under $TEXT_CACHE. Resumable: finished shards are skipped.
  4. `windows` / `agentic_batch`: windows of L tokens cut from single trajectories. Only when the
     sampled trajectory is shorter than L are trajectories concatenated, joined by SEP (= BOS).
  5. Pointer-regime labels per position (src/pointer.py): longest earlier match within the window
     over the ladder 1..64, regime none / short (1-7) / long (8+), for loss breakdowns.

Token ids are shifted by SHIFT = 2 for every tokenizer (0 = PAD, 1 = BOS), as in src/text.py.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pointer import REGIME_LONG, REGIME_NONE, REGIME_SHORT, match_lengths, regime
from .tasks import BOS, IGNORE, PT_PTR_LONG, PT_PTR_NONE, PT_PTR_SHORT

SHIFT = 2  # same as src/text.py
SEP = BOS  # joins trajectories inside a window (only when one is shorter than the window)
DATASET = "nebius/SWE-rebench-openhands-trajectories"
HF_PARQUET = "https://huggingface.co/api/datasets/{dataset}/parquet/default/train/{i}.parquet"
HF_TOKENIZER = "https://huggingface.co/{repo}/resolve/main/tokenizer.json"
COLUMNS = ["trajectory_id", "repo", "trajectory"]
SPLIT_SALT = "agentic-repo-split-v1"
SPLITS = ("train", "eval")
SPLIT_ALIAS = {"train": "train", "eval": "eval", "validation": "eval", "test": "eval"}
REGIME_TO_PT = {REGIME_NONE: PT_PTR_NONE, REGIME_SHORT: PT_PTR_SHORT, REGIME_LONG: PT_PTR_LONG}


def cache_dir() -> Path:
    """Read at call time (unlike text.CACHE), so tests and jobs can redirect it."""
    return Path(os.environ.get("TEXT_CACHE", "/tmp/locker_text_cache"))


def cpu_threads() -> int:
    return int(os.environ.get("RHOMBIC_CPU_THREADS") or os.cpu_count() or 1)


# ---------------------------------------------------------------------------------------------
# 1. Rendering


def _messages(trajectory):
    if isinstance(trajectory, dict):
        trajectory = trajectory["trajectory"]
    return trajectory or []


def _text(content) -> list[str]:
    if isinstance(content, str):
        return [content] if content else []
    if isinstance(content, list):  # defensive: list-of-blocks content (not seen in this dataset)
        return [b.get("text", "") for b in content if isinstance(b, dict) and b.get("text")]
    return []


def render_parts(trajectory, system: bool = False) -> list[str]:
    """Message pieces in order. `trajectory` is a row dict (with key "trajectory") or the message
    list. Tool-call `arguments` (a JSON string) is parsed and re-dumped, so formatting is
    canonical; unparseable arguments are kept as the raw string. Empty strings are skipped."""
    parts = []
    for m in _messages(trajectory):
        if m.get("role") == "system" and not system:
            continue
        parts.extend(_text(m.get("content")))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            parts.append(json.dumps({"name": fn.get("name"), "arguments": args}, ensure_ascii=False))
    return parts


def render(trajectory, system: bool = False) -> str:
    """Deterministic text of one trajectory: pieces joined by newlines (as pointer_stats joins
    transcript_text)."""
    return "\n".join(render_parts(trajectory, system=system))


# ---------------------------------------------------------------------------------------------
# 2. Repo-disjoint split


def repo_split(repo: str, eval_frac: float = 0.05, salt: str = SPLIT_SALT) -> str:
    """'eval' for a stable ~eval_frac of repositories, else 'train'. Depends only on the repo
    name (sha256), so it is the same on every host and for every tokenizer."""
    h = int.from_bytes(hashlib.sha256(f"{salt}:{repo}".encode()).digest()[:8], "big")
    return "eval" if h / 2.0**64 < eval_frac else "train"


# ---------------------------------------------------------------------------------------------
# 3. Tokenizers


class Tokenizer:
    """Pluggable tokenizer. ids returned shifted by SHIFT; vocab = n_vocab + SHIFT."""

    name: str
    n_vocab: int

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        raise NotImplementedError

    @property
    def vocab(self) -> int:
        return self.n_vocab + SHIFT

    @property
    def dtype(self):
        return np.uint16 if self.vocab <= 1 << 16 else np.uint32


class TiktokenGPT2(Tokenizer):
    """Our default path (src/text.py): tiktoken gpt2, encode_ordinary (no special tokens)."""

    def __init__(self):
        import tiktoken

        self.enc = tiktoken.get_encoding("gpt2")
        self.name, self.n_vocab = "gpt2", self.enc.n_vocab

    def encode_batch(self, texts):
        ids = self.enc.encode_ordinary_batch(texts, num_threads=max(1, min(32, cpu_threads())))
        return [np.asarray(d, dtype=np.int64) + SHIFT for d in ids]


class HFTokenizer(Tokenizer):
    """A Hugging Face `tokenizer.json` (the `tokenizers` library), no special tokens added."""

    def __init__(self, path: str | Path, name: str | None = None):
        os.environ.setdefault("RAYON_NUM_THREADS", str(cpu_threads()))
        from tokenizers import Tokenizer as _T

        self.tok = _T.from_file(str(path))
        self.name = name or Path(path).stem
        self.n_vocab = self.tok.get_vocab_size(with_added_tokens=True)

    def encode_batch(self, texts):
        encs = self.tok.encode_batch(texts, add_special_tokens=False)
        return [np.asarray(e.ids, dtype=np.int64) + SHIFT for e in encs]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def get_tokenizer(spec: str = "gpt2") -> Tokenizer:
    """'gpt2' -> tiktoken GPT-2; a path to a tokenizer.json -> that file; otherwise a Hugging Face
    model repo id (optionally 'hf:'-prefixed, e.g. 'm-a-p/1.3B-100B-GatedDeltaNet-pure'), whose
    tokenizer.json is downloaded once into $TEXT_CACHE/tokenizers/."""
    if spec == "gpt2":
        return TiktokenGPT2()
    if spec.endswith(".json") and Path(spec).exists():
        return HFTokenizer(spec, name=_slug(str(Path(spec).parent.name) + "-" + Path(spec).stem))
    repo = spec.removeprefix("hf:")
    path = cache_dir() / "tokenizers" / _slug(repo) / "tokenizer.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(HF_TOKENIZER.format(repo=repo)) as r:
            data = r.read()
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return HFTokenizer(path, name=_slug(repo))


def vocab_size(tokenizer: str = "gpt2") -> int:
    return get_tokenizer(tokenizer).vocab


# ---------------------------------------------------------------------------------------------
# Parquet streaming


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP Range requests, so pyarrow can read the parquet footer
    and single row groups without downloading the whole file. The HF API URL redirects to a signed
    CDN URL; that URL is resolved once and re-resolved if it expires (403/410). Transient errors
    are retried with backoff."""

    def __init__(self, url: str, retries: int = 5):
        self.src, self.retries, self.pos = url, retries, 0
        self._resolve()

    def _resolve(self):
        with urllib.request.urlopen(urllib.request.Request(self.src, headers={"Range": "bytes=0-0"})) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries:
                    raise
                if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 410):
                    self._resolve()  # signed URL expired
                time.sleep(2**attempt)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def parquet_sources(dataset: str = DATASET) -> list[str]:
    """Parquet files to read: $AGENTIC_PARQUET (comma-separated local paths or URLs) if set, else
    the HF API listing for the default/train split (one 2.1 GB file with 17 row groups of 4096
    rows as of 2026-11-07)."""
    env = os.environ.get("AGENTIC_PARQUET")
    if env:
        return [s for s in env.split(",") if s]
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{dataset}/parquet") as r:
            return list(json.load(r)["default"]["train"])
    except Exception:  # listing unavailable: the documented single-file URL
        return [HF_PARQUET.format(dataset=dataset, i=0)]


def open_parquet(src: str):
    import pyarrow.parquet as pq

    if src.startswith(("http://", "https://")):
        return pq.ParquetFile(io.BufferedReader(HTTPRangeFile(src), buffer_size=8 << 20))
    return pq.ParquetFile(src)


# ---------------------------------------------------------------------------------------------
# Build + cache


def cache_key(tokenizer: str, eval_frac: float, system: bool, max_rows: int | None) -> str:
    tok = "gpt2" if tokenizer == "gpt2" else _slug(tokenizer.removeprefix("hf:"))
    return f"agentic-{_slug(DATASET.split('/')[1])}-{tok}-sys{int(system)}-ev{eval_frac:g}-n{max_rows or 'all'}"


def build(tokenizer: str = "gpt2", *, eval_frac: float = 0.05, system: bool = False,
          max_rows: int | None = None, sources: list[str] | None = None, batch_rows: int = 128,
          verbose: bool = False) -> Path:
    """Tokenize the dataset into $TEXT_CACHE/<cache_key>/ and return that directory.

    One shard per (parquet file, row group): `<s>.<split>.bin` (raw token ids, tokenizer dtype)
    and `<s>.<split>.json` (per-trajectory lengths, repos, trajectory ids), then `<s>.done`.
    `meta.json` marks a finished cache. `max_rows` stops early (pilots, tests); it is part of the
    cache key."""
    d = cache_dir() / cache_key(tokenizer, eval_frac, system, max_rows)
    if (d / "meta.json").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer(tokenizer)
    sources = sources if sources is not None else parquet_sources()
    shards, rows_seen, s = [], 0, 0
    for src in sources:
        pf = None
        n_rg = None
        for rg in range(10**9):
            if max_rows is not None and rows_seen >= max_rows:
                break
            if n_rg is not None and rg >= n_rg:
                break
            name = f"{s:04d}"
            if (d / f"{name}.done").exists():  # resume
                info = json.loads((d / f"{name}.done").read_text())
                shards.append(name)
                rows_seen += info["rows"]
                n_rg = info["n_row_groups"]
                s += 1
                continue
            if pf is None:
                pf = open_parquet(src)
                n_rg = pf.metadata.num_row_groups
                if rg >= n_rg:
                    break
            out = {sp: dict(tokens=[], lengths=[], repos=[], ids=[]) for sp in SPLITS}
            rows = 0
            for batch in pf.iter_batches(batch_size=batch_rows, row_groups=[rg], columns=COLUMNS):
                recs = batch.to_pylist()
                if max_rows is not None:
                    recs = recs[: max_rows - rows_seen - rows]
                texts = [render(r, system=system) for r in recs]
                for r, ids in zip(recs, tok.encode_batch(texts)):
                    o = out[repo_split(r["repo"], eval_frac)]
                    o["tokens"].append(ids.astype(tok.dtype))
                    o["lengths"].append(len(ids))
                    o["repos"].append(r["repo"])
                    o["ids"].append(r["trajectory_id"])
                rows += len(recs)
                if max_rows is not None and rows_seen + rows >= max_rows:
                    break
            for sp, o in out.items():
                arr = np.concatenate(o["tokens"]) if o["tokens"] else np.zeros(0, dtype=tok.dtype)
                tmp = d / f"{name}.{sp}.bin.tmp"
                arr.tofile(tmp)
                os.replace(tmp, d / f"{name}.{sp}.bin")
                idx = dict(lengths=o["lengths"], repos=o["repos"], ids=o["ids"])
                (d / f"{name}.{sp}.json").write_text(json.dumps(idx))
            (d / f"{name}.done").write_text(json.dumps(dict(rows=rows, src=src, row_group=rg, n_row_groups=n_rg)))
            shards.append(name)
            rows_seen += rows
            s += 1
            if verbose:
                print(f"agentic build: shard {name} ({rows} rows, {rows_seen} total)", flush=True)
    counts = {}
    for sp in SPLITS:
        lens, repos = [], set()
        for name in shards:
            idx = json.loads((d / f"{name}.{sp}.json").read_text())
            lens += idx["lengths"]
            repos.update(idx["repos"])
        counts[sp] = dict(trajectories=len(lens), tokens=int(sum(lens)), repos=len(repos),
                          median_len=float(np.median(lens)) if lens else 0.0)
    meta = dict(dataset=DATASET, tokenizer=tok.name, tokenizer_spec=tokenizer, vocab=tok.vocab,
                dtype=np.dtype(tok.dtype).name, shift=SHIFT, eval_frac=eval_frac, salt=SPLIT_SALT,
                system=system, max_rows=max_rows, rows=rows_seen, shards=shards, splits=counts)
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, d / "meta.json")
    return d


class Corpus:
    """One split of a built cache: trajectory i is `get(i)` (memmapped, shifted ids)."""

    def __init__(self, d: Path, split: str):
        split = SPLIT_ALIAS[split]
        self.dir, self.split = Path(d), split
        self.meta = json.loads((self.dir / "meta.json").read_text())
        dtype = np.dtype(self.meta["dtype"])
        self._arrays, shard_of, starts, lengths, self.repos, self.ids = [], [], [], [], [], []
        for k, name in enumerate(self.meta["shards"]):
            idx = json.loads((self.dir / f"{name}.{split}.json").read_text())
            p = self.dir / f"{name}.{split}.bin"
            n = p.stat().st_size // dtype.itemsize
            self._arrays.append(np.memmap(p, dtype=dtype, mode="r") if n else np.zeros(0, dtype=dtype))
            ln = np.asarray(idx["lengths"], dtype=np.int64)
            st = np.concatenate([[0], np.cumsum(ln)[:-1]]) if len(ln) else ln
            shard_of += [k] * len(ln)
            starts.append(st)
            lengths.append(ln)
            self.repos += idx["repos"]
            self.ids += idx["ids"]
        self.shard_of = np.asarray(shard_of, dtype=np.int64)
        self.starts = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)
        self.lengths = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64)
        self.cum = np.cumsum(self.lengths)
        self.total = int(self.cum[-1]) if len(self.cum) else 0

    def __len__(self):
        return len(self.lengths)

    def get(self, i: int) -> np.ndarray:
        a = self._arrays[self.shard_of[i]]
        return a[self.starts[i] : self.starts[i] + self.lengths[i]]


_CORPORA: dict = {}


def load(split: str = "train", tokenizer: str = "gpt2", *, eval_frac: float = 0.05, system: bool = False,
         max_rows: int | None = None) -> Corpus:
    """Corpus for a split, building the cache first if needed (memoized per process)."""
    key = (SPLIT_ALIAS[split], tokenizer, eval_frac, system, max_rows, str(cache_dir()))
    if key not in _CORPORA:
        d = build(tokenizer, eval_frac=eval_frac, system=system, max_rows=max_rows)
        _CORPORA[key] = Corpus(d, split)
    return _CORPORA[key]


# ---------------------------------------------------------------------------------------------
# 4. Windows


def sample_window(corpus: Corpus, L: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """(window int64 [L], packed). Trajectories are drawn with probability proportional to length
    (every token equally likely to be covered), the start uniformly. A window crosses trajectories
    only when the drawn one is shorter than L: then further trajectories (same draw rule) are
    appended, each preceded by SEP, and the result is cut to L (packed=True)."""
    if corpus.total == 0:
        raise ValueError(f"empty agentic split '{corpus.split}' in {corpus.dir}")

    def draw():
        return int(np.searchsorted(corpus.cum, rng.integers(corpus.total), side="right"))

    i = draw()
    n = int(corpus.lengths[i])
    if n >= L:
        s = int(rng.integers(0, n - L + 1))
        return np.asarray(corpus.get(i)[s : s + L], dtype=np.int64), False
    pieces, have = [np.asarray(corpus.get(i), dtype=np.int64)], n
    while have < L:
        j = draw()
        pieces += [np.array([SEP], dtype=np.int64), np.asarray(corpus.get(j), dtype=np.int64)]
        have += 1 + int(corpus.lengths[j])
    return np.concatenate(pieces)[:L], True


def windows(split: str, L: int, n: int, seed: int, **corpus_kw):
    """Yield n int64 arrays of length L from `split` ('train' or 'eval'), deterministic in seed
    (for a given cache). corpus_kw: tokenizer, eval_frac, system, max_rows (see `load`)."""
    corpus = load(split, **corpus_kw)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield sample_window(corpus, L, rng)[0]


# ---------------------------------------------------------------------------------------------
# 5. Pointer-regime labels


def pointer_labels(win: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(match_len int16 [L], regime int8 [L]) for a window, where position t predicts win[t] from
    win[:t]: longest earlier match within the window over the ladder 1..64 (src/pointer.py),
    regime 0 none / 1 short (1-7) / 2 long (8+)."""
    ml, _ = match_lengths(win)
    return ml.astype(np.int16), regime(ml)


# ---------------------------------------------------------------------------------------------
# 6. Task entry (make_task name="agentic", mirrors name="lm" in src/text.py)


@dataclass(frozen=True)
class AgenticConfig:
    split: str = "train"  # 'train' or 'eval' ('validation' / 'test' alias 'eval')
    L_max: int = 4096
    tokenizer: str = "gpt2"  # 'gpt2', a tokenizer.json path, or an HF model repo id
    eval_frac: float = 0.05
    system: bool = False
    max_rows: int | None = None  # cap rows read (pilots); part of the cache key


def agentic_batch(cfg: AgenticConfig, B: int, rng: np.random.Generator) -> dict:
    """Batch in the src/tasks.py format, like text.lm_batch: tokens = BOS + window[:L-1],
    targets = window. pos_type = pointer regime of the target (PT_PTR_NONE/SHORT/LONG), with
    position 0 as BOS (unscored). Extra keys: match_len int16 [B, L] (aligned with targets) and
    packed bool [B] (window crossed a trajectory boundary)."""
    corpus = load(cfg.split, cfg.tokenizer, eval_frac=cfg.eval_frac, system=cfg.system, max_rows=cfg.max_rows)
    L = cfg.L_max
    win = np.empty((B, L), dtype=np.int64)
    packed = np.zeros(B, dtype=bool)
    match_len = np.zeros((B, L), dtype=np.int16)
    pos_type = np.empty((B, L), dtype=np.int8)
    for b in range(B):
        win[b], packed[b] = sample_window(corpus, L, rng)
        match_len[b], reg = pointer_labels(win[b])
        pos_type[b] = np.choose(reg, [REGIME_TO_PT[REGIME_NONE], REGIME_TO_PT[REGIME_SHORT], REGIME_TO_PT[REGIME_LONG]])
    tokens = np.empty((B, L), dtype=np.int64)
    tokens[:, 0] = BOS
    tokens[:, 1:] = win[:, : L - 1]
    targets = win.copy()
    targets[:, 0] = IGNORE  # BOS position: no context yet
    pos_type[:, 0] = 1  # PT_BOS
    return dict(tokens=tokens, targets=targets, pos_type=pos_type,
                corrupt=np.zeros((B, L), dtype=bool), src_pos=np.full((B, L), -1, dtype=np.int64),
                match_len=match_len, packed=packed)


def prewarm(task: dict) -> Path:
    """Build the cache for a task dict (call once before DataLoader workers start)."""
    t = {k: v for k, v in task.items() if k in ("tokenizer", "eval_frac", "system", "max_rows")}
    return build(**t, verbose=True)
