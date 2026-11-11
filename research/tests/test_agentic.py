"""Agentic-trajectory data (src/agentic.py) and pointer labels (src/pointer.py). CPU, no network:
trajectories are hand-made in the dataset's schema and written to a local parquet file."""
import json

import numpy as np
import pytest

from src import agentic, pointer
from src.tasks import BOS, IGNORE, PT_NAMES, PT_PTR_LONG, PT_PTR_NONE, PT_PTR_SHORT, make_task


def _msg(role, content=None, tool_calls=None, name=None, tool_call_id=None):
    return dict(content=content, name=name, role=role, tool_call_id=tool_call_id, tool_calls=tool_calls)


def _call(name, args: dict, cid="call-1"):
    return dict(function=dict(arguments=json.dumps(args), name=name), id=cid, type="function")


def make_traj(repo: str, i: int, n_turns: int = 3, body: str = "") -> dict:
    """One row in the nebius/SWE-rebench-openhands-trajectories schema (fields we read + a few)."""
    msgs = [_msg("system", "You are OpenHands agent."),
            _msg("user", f"<issue_description>\nBug {i} in {repo}\n</issue_description>")]
    for k in range(n_turns):
        msgs.append(_msg("assistant", f"Step {k}: look at the file.",
                         tool_calls=[_call("str_replace_editor", {"command": "view", "path": f"/workspace/{repo}/m{k}.py"},
                                           cid=f"c{k}")]))
        msgs.append(_msg("tool", f"Here's the result of running `cat -n`:\n     1\tdef f{k}():\n     2\t    return {k}\n{body}",
                         name="str_replace_editor", tool_call_id=f"c{k}"))
    msgs.append(_msg("assistant", None, tool_calls=[_call("finish", {"message": "done"}, cid="cf")]))
    return dict(trajectory_id=f"t-{repo}-{i}", instance_id=f"{repo}-{i}", repo=repo, trajectory=msgs,
                model_patch="", exit_status="submit", resolved=1)


TRAJ = dict(trajectory_id="t0", repo="acme/widgets", trajectory=[
    _msg("system", "SYSTEM PROMPT"),
    _msg("user", "Fix the bug."),
    _msg("assistant", "Let me look.", tool_calls=[_call("execute_bash", {"command": "cat a.py"}, "c1")]),
    _msg("tool", "x = 1\n  y = 2\n", name="execute_bash", tool_call_id="c1"),
    _msg("assistant", None, tool_calls=[_call("think", {"thought": "naïve ok"}, "c2"),
                                        dict(function=dict(arguments="{not json", name="finish"), id="c3", type="function")]),
    _msg("tool", "", name="think", tool_call_id="c2"),
])


def test_render_content_and_determinism():
    s = agentic.render(TRAJ)
    expected = "\n".join([
        "Fix the bug.",
        "Let me look.",
        '{"name": "execute_bash", "arguments": {"command": "cat a.py"}}',
        "x = 1\n  y = 2\n",  # tool output verbatim
        '{"name": "think", "arguments": {"thought": "naïve ok"}}',  # ensure_ascii=False
        '{"name": "finish", "arguments": "{not json"}',  # unparseable arguments kept raw
    ])
    assert s == expected
    assert agentic.render(TRAJ) == s and agentic.render(TRAJ["trajectory"]) == s
    assert agentic.render(TRAJ, system=True) == "SYSTEM PROMPT\n" + s


def test_repo_split_disjoint_and_fraction():
    repos = [f"org{i % 37}/repo-{i}" for i in range(4000)]
    sp = {r: agentic.repo_split(r) for r in repos}
    assert sp == {r: agentic.repo_split(r) for r in repos}  # stable
    frac = np.mean([v == "eval" for v in sp.values()])
    assert 0.035 < frac < 0.065
    assert agentic.repo_split("a/b", eval_frac=0.0) == "train" and agentic.repo_split("a/b", eval_frac=1.0) == "eval"


@pytest.fixture
def tiny_parquet(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    monkeypatch.setenv("TEXT_CACHE", str(tmp_path / "cache"))
    agentic._CORPORA.clear()
    rows = [make_traj(f"org/r{i % 10}", i, n_turns=2 + i % 3, body="pad line\n" * (i % 5)) for i in range(40)]
    path = tmp_path / "traj.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=16)  # 3 row groups -> 3 shards
    monkeypatch.setenv("AGENTIC_PARQUET", str(path))
    yield rows
    agentic._CORPORA.clear()


def test_build_corpus_repo_disjoint_and_decodes(tiny_parquet):
    d = agentic.build("gpt2", eval_frac=0.5)
    meta = json.loads((d / "meta.json").read_text())
    assert meta["rows"] == 40 and len(meta["shards"]) == 3 and meta["dtype"] == "uint16"
    tr, ev = agentic.load("train", eval_frac=0.5), agentic.load("eval", eval_frac=0.5)
    assert len(tr) + len(ev) == 40 and len(tr) > 0 and len(ev) > 0
    assert not set(tr.repos) & set(ev.repos)
    assert all(agentic.repo_split(r, 0.5) == "eval" for r in ev.repos)
    enc = agentic.get_tokenizer("gpt2").enc
    by_id = {r["trajectory_id"]: r for r in tiny_parquet}
    for c in (tr, ev):
        for i in range(len(c)):
            assert enc.decode((np.asarray(c.get(i), dtype=np.int64) - agentic.SHIFT).tolist()) == agentic.render(by_id[c.ids[i]])
    # A rebuild from finished shards (meta removed) is identical.
    first = (d / "0000.train.bin").read_bytes()
    (d / "meta.json").unlink()
    assert agentic.build("gpt2", eval_frac=0.5) == d and (d / "0000.train.bin").read_bytes() == first
    # max_rows caps the rows read and has its own cache entry.
    d5 = agentic.build("gpt2", eval_frac=0.5, max_rows=5)
    assert d5 != d and json.loads((d5 / "meta.json").read_text())["rows"] == 5


def test_windows_shapes_determinism_and_single_trajectory(tiny_parquet):
    kw = dict(eval_frac=0.5)
    c = agentic.load("train", **kw)
    L = 32
    assert c.lengths.min() > L  # every trajectory is longer than the window: no packing
    a = list(agentic.windows("train", L, 6, seed=3, **kw))
    b = list(agentic.windows("train", L, 6, seed=3, **kw))
    other = list(agentic.windows("train", L, 6, seed=4, **kw))
    assert all(w.shape == (L,) and w.dtype == np.int64 for w in a)
    assert all((x == y).all() for x, y in zip(a, b)) and any((x != y).any() for x, y in zip(a, other))
    trajs = [np.asarray(c.get(i), dtype=np.int64) for i in range(len(c))]
    for w in a:  # each window is a contiguous span of one trajectory
        assert BOS not in w
        assert any(any((t[s:s + L] == w).all() for s in range(len(t) - L + 1)) for t in trajs)


def test_short_trajectories_are_packed_with_separator(tiny_parquet):
    c = agentic.load("train", eval_frac=0.5)
    L = int(c.lengths.max()) * 3
    rng = np.random.default_rng(0)
    for _ in range(4):
        w, packed = agentic.sample_window(c, L, rng)
        assert packed and w.shape == (L,) and (w == agentic.SEP).sum() >= 2


def _loop_match_lengths(tok):
    """The original scripts/pointer_stats.analyze loop (binary search over the ladder)."""
    T, LADDER = len(tok), pointer.LADDER
    keys = pointer.suffix_keys(tok)
    last = {L: {} for L in LADDER}
    ml, src = np.zeros(T, dtype=np.int64), np.full(T, -1, dtype=np.int64)
    for t in range(1, T):
        lo, hi, best = 0, len(LADDER) - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            L = LADDER[mid]
            if t >= L and int(keys[L][t]) in last[L]:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        if best >= 0:
            L = LADDER[best]
            ml[t], src[t] = L, last[L][int(keys[L][t])] + 1
        for L in LADDER:
            if t >= L:
                last[L][int(keys[L][t])] = t - 1
    return ml, src


def test_suffix_keys_match_prefix_hash():
    tok = np.random.default_rng(0).integers(0, 1000, 300)
    x = tok.astype(np.uint64) + np.uint64(1)
    H = np.zeros(len(tok) + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(len(tok)):
            H[i + 1] = H[i] * pointer.P + x[i]
        keys = pointer.suffix_keys(tok)
        for L in (1, 8, 64):
            want = H[L:] - H[:-L] * (pointer.P ** np.uint64(L))
            assert (keys[L][L:] == (want ^ np.uint64(L * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF))).all()


@pytest.mark.parametrize("alphabet", [3, 20, 5000])
def test_match_lengths_equal_loop_reference(alphabet):
    rng = np.random.default_rng(alphabet)
    tok = rng.integers(0, alphabet, 600)
    tok[400:500] = tok[100:200]  # a long copy
    ml, src = pointer.match_lengths(tok)
    ml2, src2 = _loop_match_lengths(tok)
    assert (ml == ml2).all() and (src == src2).all()


def test_pointer_labels_on_synthetic_repeat():
    n, r = 100, 80
    head = np.random.default_rng(0).permutation(10_000)[:n] + 2  # distinct tokens: no chance matches
    win = np.concatenate([head, head[:r]])
    ml, reg = agentic.pointer_labels(win)
    assert (ml[: n + 1] == 0).all()  # nothing repeats before the copy; at t=n the suffix is head[n-1]
    for k in range(1, r):
        want = max(L for L in pointer.LADDER if L <= k)  # suffix head[:k] matches, capped by the ladder
        assert ml[n + k] == want, (k, ml[n + k], want)
        assert reg[n + k] == (1 if want <= 7 else 2)
    _, src = pointer.match_lengths(win)
    assert (src[n + 1 : n + r] == np.arange(1, r)).all()  # pointer points at the copied token
    assert (reg[: n + 1] == pointer.REGIME_NONE).all()

def test_gpt2_tokenizer_plumbing():
    t = agentic.get_tokenizer("gpt2")
    ids = t.encode_batch(["hello world", ""])
    assert t.vocab == t.enc.n_vocab + agentic.SHIFT and t.dtype == np.uint16
    assert ids[0].min() >= agentic.SHIFT and len(ids[1]) == 0
    assert t.enc.decode((ids[0] - agentic.SHIFT).tolist()) == "hello world"


@pytest.fixture
def tiny_hf_tokenizer(tmp_path):
    tk = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers, trainers

    tok = tk.Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = [agentic.render(make_traj(f"org/r{i}", i)) for i in range(20)]
    tok.train_from_iterator(corpus, trainer)
    from tokenizers import decoders

    tok.decoder = decoders.ByteLevel()
    path = tmp_path / "tok" / "tokenizer.json"
    path.parent.mkdir()
    tok.save(str(path))
    return path


def test_hf_tokenizer_plumbing(tiny_hf_tokenizer):
    t = agentic.get_tokenizer(str(tiny_hf_tokenizer))
    s = agentic.render(TRAJ)
    ids = t.encode_batch([s])[0]
    assert t.n_vocab <= 300 and t.vocab == t.n_vocab + agentic.SHIFT
    assert ids.min() >= agentic.SHIFT and ids.max() < t.vocab
    assert t.tok.decode((ids - agentic.SHIFT).tolist()) == s
    assert agentic.vocab_size(str(tiny_hf_tokenizer)) == t.vocab


def test_agentic_batch_via_make_task(tiny_parquet, tiny_hf_tokenizer):
    task = dict(name="agentic", L_max=48, tokenizer=str(tiny_hf_tokenizer), eval_frac=0.5)
    cfg, fn = make_task(task)
    b = fn(4, np.random.default_rng(0))
    tok, tgt, pt = b["tokens"], b["targets"], b["pos_type"]
    assert tok.shape == tgt.shape == pt.shape == b["match_len"].shape == (4, 48)
    assert (tok[:, 0] == BOS).all() and (tgt[:, 0] == IGNORE).all() and (tgt[:, 1:-1] == tok[:, 2:]).all()
    assert set(np.unique(pt[:, 1:])) <= {PT_PTR_NONE, PT_PTR_SHORT, PT_PTR_LONG}
    assert PT_NAMES[PT_PTR_LONG] == "ptr_long"
    for i in range(4):  # labels are the pointer regime of each target within its window
        win = np.concatenate([tok[i, 1:2], tgt[i, 1:]])  # the sampled window
        ml, reg = agentic.pointer_labels(win)
        assert (b["match_len"][i, 1:] == ml[1:]).all()
        assert (pt[i, 1:] == np.array([PT_PTR_NONE, PT_PTR_SHORT, PT_PTR_LONG])[reg[1:]]).all()
    _, efn = make_task({**task, "split": "validation"})  # train.py's eval path -> held-out repos
    assert efn(2, np.random.default_rng(1))["tokens"].shape == (2, 48)


def test_agentic_training_end_to_end(tmp_path, tiny_parquet, tiny_hf_tokenizer):
    from src.train import run

    cfg = dict(run_id="ag", arm="ovl", task=dict(name="agentic", L_max=64, tokenizer=str(tiny_hf_tokenizer), eval_frac=0.5),
               steps=2, eval_every=2, eval_size=8, batch=4, num_workers=0, overrides={"mixer": "attn"})
    final = run(cfg, tmp_path)
    assert final["acc"] is not None
    rec = [json.loads(l) for l in (tmp_path / "ag.jsonl").read_text().splitlines() if '"step"' in l][0]
    assert {"ptr_long", "ptr_short"} <= set(rec["loss_by_type"])
