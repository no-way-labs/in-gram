"""Task generators (plan.md §3).

Every batch is a dict of numpy arrays:
  tokens   int64 [B, L]   model input (teacher forcing)
  targets  int64 [B, L]   next-token target at scored positions, IGNORE elsewhere
  pos_type int8  [B, L]   role of each position (PT_* below), used by diagnostics
  corrupt  bool  [B, L]   at scored positions: query key was corrupted
  src_pos  int64 [B, L]   at scored positions: context position of the value
                          token holding the answer (-1 if n/a)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

VOCAB = 320
PAD, BOS = 0, 1
KEY0, N_KEY = 2, 64
VAL0, N_VAL = 66, 64
NOISE0, N_NOISE = 130, 128
DIGIT0 = 258
IGNORE = -100

PT_PAD, PT_BOS, PT_KEY, PT_VAL, PT_NOISE, PT_QKEY, PT_QFINAL, PT_QANS, PT_DIGIT, PT_SUM = range(10)
PT_NAMES = ["pad", "bos", "key", "val", "noise", "qkey", "qfinal", "qans", "digit", "sum",
            # 10-13: real-text positions (src/text.py). Code 11 was "text_rep" (all repeats) in
            # lm-r1..r3; from phase 2 it is repeats whose previous occurrence is < 256 tokens back.
            "text_novel", "text_rep_near", "text_rep_mid", "text_rep_far",
            # 14-16: agentic trajectories (src/agentic.py), the target's pointer regime: longest
            # earlier in-window match none / 1-7 / 8+ tokens (src/pointer.py ladder 1..64).
            "ptr_none", "ptr_short", "ptr_long"]
PT_PTR_NONE, PT_PTR_SHORT, PT_PTR_LONG = 14, 15, 16


def round_up(x: int, m: int) -> int:
    return -(-x // m) * m


@dataclass(frozen=True)
class KVConfig:
    k: int = 2
    key_alphabet: int = 16
    D: int = 64
    C: int = 384
    Q: int = 32
    L_max: int = 512
    corrupt: float = 0.0
    rebind: float = 0.0
    rebinds: int = 1  # later bindings per rebound key; each value differs from all earlier ones

    def validate(self) -> None:
        n_reb = round(self.rebind * self.D) * self.rebinds
        if not 1 <= self.rebinds < N_VAL:
            raise ValueError("rebinds must be in [1, N_VAL)")
        if self.key_alphabet > N_KEY:
            raise ValueError("key_alphabet > 64")
        if self.D > self.key_alphabet ** self.k:
            raise ValueError("D exceeds number of possible keys")
        if self.Q > self.D:
            raise ValueError("Q > D")
        if self.C < (self.D + n_reb) * (self.k + 1):
            raise ValueError(f"C={self.C} < (D + n_rebound)*(k+1)={(self.D + n_reb) * (self.k + 1)}")
        if 1 + self.C + self.Q * (self.k + 1) > self.L_max:
            raise ValueError("sequence does not fit in L_max")
        if round(self.corrupt * self.Q) > 0 and self.D >= self.key_alphabet ** self.k:
            raise ValueError("corruption needs unused keys")


def kv_batch(cfg: KVConfig, B: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    cfg.validate()
    k, a, D, C, Q, L = cfg.k, cfg.key_alphabet, cfg.D, cfg.C, cfg.Q, cfg.L_max
    n_reb = round(cfg.rebind * D) * cfg.rebinds
    n_cor = round(cfg.corrupt * Q)
    m = D + n_reb  # number of (key, value) items placed in context

    tokens = np.full((B, L), PAD, dtype=np.int64)
    targets = np.full((B, L), IGNORE, dtype=np.int64)
    pos_type = np.full((B, L), PT_PAD, dtype=np.int8)
    corrupt = np.zeros((B, L), dtype=bool)
    src_pos = np.full((B, L), -1, dtype=np.int64)
    powers = a ** np.arange(k - 1, -1, -1)

    for b in range(B):
        key_ids = rng.choice(a ** k, size=D, replace=False)
        keys = KEY0 + (key_ids[:, None] // powers) % a  # [D, k]
        vals = rng.integers(VAL0, VAL0 + N_VAL, size=D)

        # Items: originals (key j, value vals[j]) and rebinds (key j, new value).
        if cfg.rebinds == 1:  # original RNG stream, kept so logged runs reproduce
            item_key = np.concatenate([np.arange(D), rng.choice(D, size=n_reb, replace=False)])
            item_val = np.concatenate([vals, np.zeros(n_reb, dtype=np.int64)])
            for i in range(D, m):
                j = item_key[i]
                v = rng.integers(VAL0, VAL0 + N_VAL - 1)
                item_val[i] = v + (v >= vals[j])  # different from the original value
        else:  # each rebound key gets `rebinds` new values, all distinct from each other and vals[j]
            R = cfg.rebinds
            reb_keys = rng.choice(D, size=n_reb // R, replace=False)
            item_key = np.concatenate([np.arange(D), np.repeat(reb_keys, R)])
            new_vals = []
            for j in reb_keys:
                pool = np.setdiff1d(np.arange(VAL0, VAL0 + N_VAL), [vals[j]])
                new_vals.append(rng.choice(pool, size=R, replace=False))
            item_val = np.concatenate([vals] + new_vals).astype(np.int64)

        # Non-overlapping random placement within context positions 1..C.
        free = C - m * (k + 1)
        starts = np.sort(rng.choice(free + m, size=m, replace=False)) + np.arange(m) * k + 1
        slot_of_item = rng.permutation(m)  # item i goes to starts[slot_of_item[i]]

        # For a rebound key, bindings appear in item order (original first, then each rebind).
        if cfg.rebinds == 1:
            for i in range(D, m):
                j = item_key[i]
                if slot_of_item[i] < slot_of_item[j]:
                    slot_of_item[i], slot_of_item[j] = slot_of_item[j], slot_of_item[i]
        else:
            for j in np.unique(item_key[D:]):
                idx = np.nonzero(item_key == j)[0]  # ascending item order
                slot_of_item[idx] = np.sort(slot_of_item[idx])

        tokens[b, 0] = BOS
        pos_type[b, 0] = PT_BOS
        tokens[b, 1 : C + 1] = rng.integers(NOISE0, NOISE0 + N_NOISE, size=C)
        pos_type[b, 1 : C + 1] = PT_NOISE
        latest_pos = np.full(D, -1)
        latest_val = np.zeros(D, dtype=np.int64)
        for i in range(m):
            s = starts[slot_of_item[i]]
            j = item_key[i]
            tokens[b, s : s + k] = keys[j]
            tokens[b, s + k] = item_val[i]
            pos_type[b, s : s + k] = PT_KEY
            pos_type[b, s + k] = PT_VAL
            if s + k > latest_pos[j]:
                latest_pos[j] = s + k
                latest_val[j] = item_val[i]

        key_set = set(key_ids.tolist())
        qkeys = rng.choice(D, size=Q, replace=False)
        cor_q = set(rng.choice(Q, size=n_cor, replace=False).tolist())
        for q, j in enumerate(qkeys):
            s = 1 + C + q * (k + 1)
            key = keys[j].copy()
            if q in cor_q:
                while True:
                    cand = key.copy()
                    i = rng.integers(k)
                    t = rng.integers(a - 1)
                    cur = cand[i] - KEY0
                    cand[i] = KEY0 + t + (t >= cur)
                    if int(((cand - KEY0) * powers).sum()) not in key_set:
                        key = cand
                        break
            tokens[b, s : s + k] = key
            tokens[b, s + k] = latest_val[j]
            pos_type[b, s : s + k - 1] = PT_QKEY
            pos_type[b, s + k - 1] = PT_QFINAL
            pos_type[b, s + k] = PT_QANS
            targets[b, s + k - 1] = latest_val[j]
            corrupt[b, s + k - 1] = q in cor_q
            src_pos[b, s + k - 1] = latest_pos[j]

    return dict(tokens=tokens, targets=targets, pos_type=pos_type, corrupt=corrupt, src_pos=src_pos)


TOPIC0 = 268  # topic marker tokens live in the unused range 268..319


@dataclass(frozen=True)
class TopicKVConfig:
    """Homograph recall: T topic sections, each opened by a topic token; every key is bound in
    EVERY section to a different value. A query is `topic key -> value of key in that topic`.
    The key N-gram is identical across topics, so exact hashing alone cannot pick the value;
    context (which topic) must."""
    k: int = 2
    key_alphabet: int = 16
    D: int = 32
    T: int = 2
    C_t: int = 192  # positions per topic section (after its topic token)
    Q: int = 32
    L_max: int = 576
    # Shuffle the order of topic sections per sequence. Without it, "earlier write = topic A" is
    # a fixed rule a delta-rule superposition can exploit (r5-homograph). On by default.
    shuffle_topics: bool = True

    def validate(self) -> None:
        if self.T > 52 or self.D > self.key_alphabet ** self.k or self.Q > self.D * self.T:
            raise ValueError("bad topic_kv sizes")
        if self.C_t < self.D * (self.k + 1):
            raise ValueError("C_t too small for D pairs")
        if 1 + self.T * (1 + self.C_t) + self.Q * (self.k + 2) > self.L_max:
            raise ValueError("sequence does not fit in L_max")


def topic_kv_batch(cfg: TopicKVConfig, B: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    cfg.validate()
    k, a, D, T, Ct, Q, L = cfg.k, cfg.key_alphabet, cfg.D, cfg.T, cfg.C_t, cfg.Q, cfg.L_max
    tokens = np.full((B, L), PAD, dtype=np.int64)
    targets = np.full((B, L), IGNORE, dtype=np.int64)
    pos_type = np.full((B, L), PT_PAD, dtype=np.int8)
    src_pos = np.full((B, L), -1, dtype=np.int64)
    powers = a ** np.arange(k - 1, -1, -1)
    free = Ct - D * (k + 1)
    for b in range(B):
        keys = KEY0 + (rng.choice(a ** k, size=D, replace=False)[:, None] // powers) % a
        vals = np.stack([rng.permutation(N_VAL)[:T] for _ in range(D)]) + VAL0  # [D, T], distinct per key
        tokens[b, 0], pos_type[b, 0] = BOS, PT_BOS
        val_pos = np.zeros((D, T), dtype=np.int64)
        order = rng.permutation(T) if cfg.shuffle_topics else np.arange(T)  # topic of each section
        sec_of = np.empty(T, dtype=np.int64)
        sec_of[order] = np.arange(T)  # section index holding topic t
        for t in range(T):
            base = 1 + sec_of[t] * (1 + Ct)
            tokens[b, base], pos_type[b, base] = TOPIC0 + t, PT_KEY  # topic marker
            tokens[b, base + 1 : base + 1 + Ct] = rng.integers(NOISE0, NOISE0 + N_NOISE, size=Ct)
            pos_type[b, base + 1 : base + 1 + Ct] = PT_NOISE
            starts = np.sort(rng.choice(free + D, size=D, replace=False)) + np.arange(D) * k + base + 1
            for j, st in zip(rng.permutation(D), starts):
                tokens[b, st : st + k], tokens[b, st + k] = keys[j], vals[j, t]
                pos_type[b, st : st + k], pos_type[b, st + k] = PT_KEY, PT_VAL
                val_pos[j, t] = st + k
        q0 = 1 + T * (1 + Ct)
        picks = rng.choice(D * T, size=Q, replace=False)
        for q, pk in enumerate(picks):
            j, t = divmod(int(pk), T)
            s = q0 + q * (k + 2)
            tokens[b, s], pos_type[b, s] = TOPIC0 + t, PT_QKEY
            tokens[b, s + 1 : s + 1 + k] = keys[j]
            pos_type[b, s + 1 : s + k] = PT_QKEY
            pos_type[b, s + k] = PT_QFINAL
            tokens[b, s + k + 1], pos_type[b, s + k + 1] = vals[j, t], PT_QANS
            targets[b, s + k] = vals[j, t]
            src_pos[b, s + k] = val_pos[j, t]
    return dict(tokens=tokens, targets=targets, pos_type=pos_type, corrupt=np.zeros((B, L), dtype=bool),
                src_pos=src_pos)


@dataclass(frozen=True)
class ModSumConfig:
    # 255 digits (not 256): BOS + 2*256 tokens would not fit in L_max=512. See DEVIATIONS.md D3.
    n_digits: int = 255
    L_max: int = 512


def modsum_batch(cfg: ModSumConfig, B: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n, L = cfg.n_digits, cfg.L_max
    assert 1 + 2 * n <= L
    d = rng.integers(0, 10, size=(B, n))
    s = np.cumsum(d, axis=1) % 10
    tokens = np.full((B, L), PAD, dtype=np.int64)
    targets = np.full((B, L), IGNORE, dtype=np.int64)
    pos_type = np.full((B, L), PT_PAD, dtype=np.int8)
    tokens[:, 0] = BOS
    pos_type[:, 0] = PT_BOS
    tokens[:, 1 : 2 * n + 1 : 2] = DIGIT0 + d
    tokens[:, 2 : 2 * n + 2 : 2] = DIGIT0 + s
    pos_type[:, 1 : 2 * n + 1 : 2] = PT_DIGIT
    pos_type[:, 2 : 2 * n + 2 : 2] = PT_SUM
    targets[:, 1 : 2 * n + 1 : 2] = DIGIT0 + s  # predict s_i at the position of d_i
    return dict(
        tokens=tokens,
        targets=targets,
        pos_type=pos_type,
        corrupt=np.zeros((B, L), dtype=bool),
        src_pos=np.full((B, L), -1, dtype=np.int64),
    )


def make_task(task: dict):
    """task dict -> (config, batch_fn(B, rng))."""
    task = dict(task)
    name = task.pop("name", "kv")
    if name == "kv":
        cfg = KVConfig(**task)
        cfg.validate()
        return cfg, lambda B, rng: kv_batch(cfg, B, rng)
    if name == "modsum":
        cfg = ModSumConfig(**task)
        return cfg, lambda B, rng: modsum_batch(cfg, B, rng)
    if name == "topic_kv":
        cfg = TopicKVConfig(**task)
        cfg.validate()
        return cfg, lambda B, rng: topic_kv_batch(cfg, B, rng)
    if name == "lm":
        from .text import LMConfig, lm_batch

        cfg = LMConfig(**task)
        return cfg, lambda B, rng: lm_batch(cfg, B, rng)
    if name == "agentic":
        from .agentic import AgenticConfig, agentic_batch

        cfg = AgenticConfig(**task)
        return cfg, lambda B, rng: agentic_batch(cfg, B, rng)
    raise ValueError(f"unknown task {name}")


def task_dict(cfg) -> dict:
    d = asdict(cfg)
    d["name"] = "kv" if isinstance(cfg, KVConfig) else "modsum"
    return d
