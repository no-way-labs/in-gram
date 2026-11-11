"""Arms and controls (plan.md §6) -> (model cfg, schedule cfg).

Overrides: any key in `overrides` replaces the model cfg (top level) and the
`overlay` sub-dict keys via `overlay.<key>`, and schedule keys via `sched.<key>`.
This is how diagnostic variants are expressed without new arm names.
"""
from __future__ import annotations

import copy

FROZEN_ROWS = 1 << 16


def arm_config(arm: str, task: dict, overrides: dict | None = None) -> tuple[dict, dict]:
    k = task.get("k", 2)
    L = task.get("L_max", 512)
    base = dict(mixer="gdn", d_model=128, n_blocks=2, heads=2, mlp_hidden=256, max_len=L, overlay=None)
    ovl = dict(kind="memory", orders=(k,), H=2, d_mem=32)
    sched = dict(orders=(k,), H=2, mode="growing", M=None, ways=4, read="normal", table_rows=None)
    m, s = copy.deepcopy(base), dict(sched)

    if arm == "attn":
        m["mixer"] = "attn"
    elif arm == "base":
        pass
    elif arm == "base_param":
        m["mlp_hidden"] = None  # resolved by match_params() at build time
    elif arm == "base_state":
        m["heads"] = None  # resolved by match_state() at build time
    elif arm == "ovl":
        m["overlay"] = ovl
    elif arm == "ovl_bounded":
        m["overlay"] = ovl
        s.update(mode="bounded", M=64)
    elif arm == "ovl_multi":
        m["overlay"] = dict(ovl, orders=(1, 2, 3))
        s["orders"] = (1, 2, 3)
    elif arm == "ctrl_random":
        m["overlay"] = ovl
        s["read"] = "random"
    elif arm == "ctrl_frozen":
        m["overlay"] = dict(kind="frozen", orders=(k,), H=2, d_mem=32, rows=FROZEN_ROWS)
        s["table_rows"] = FROZEN_ROWS
    elif arm == "locker_gdn":  # block 1 = locker-only layer, block 2 = GDN
        m["mixer"] = ["locker", "gdn"]
        m["overlay"] = ovl
    elif arm == "sdm_gdn":  # block 1 = Sparse Delta Memory (learned addressing), block 2 = GDN
        m["mixer"] = ["sdm", "gdn"]
    elif arm in ("ngh", "ngh123"):  # n-gram head baseline (Akyurek et al. 2024)
        m["overlay"] = dict(kind="memory", orders=(k,) if arm == "ngh" else (1, 2, 3), H=1, d_mem=32,
                            write_mode="mean", gate=False, empty_flag=False)
    elif arm == "hgdn":  # Gated DeltaNet with hashed one-hot cue columns per head (src/hgdn.py)
        m["mixer"] = "hgdn"
        m["hash"] = dict(orders=(1, 2, 3, 4, 5), write_mode="mean", couple=False)
    elif arm == "ovl_only":
        m["mixer"] = "none"
        m["overlay"] = ovl
    else:
        raise ValueError(f"unknown arm {arm}")

    for key, val in (overrides or {}).items():
        if key.startswith("overlay."):
            m["overlay"][key.split(".", 1)[1]] = val
        elif key.startswith("hash."):
            m["hash"][key.split(".", 1)[1]] = val
        elif key.startswith("sched."):
            s[key.split(".", 1)[1]] = val
        else:
            m[key] = val
    if m["overlay"] and m["overlay"].get("prior_rows"):  # Bayesian locker reads hashed prior rows
        s["table_rows"] = int(m["overlay"]["prior_rows"])
    rows = m.pop("static_rows", None)
    if rows:  # stack a static hashed table (Engram-style) under the arm's memory
        if not m["overlay"]:  # static only: cues from the sched.* overrides
            s["orders"] = tuple(o if isinstance(o, int) else tuple(o) for o in s["orders"])
    if m["overlay"] and "orders" in m["overlay"]:
        # JSON gives lists; skip-gram patterns are tuples of offsets (see hashing.pattern_offsets).
        norm = tuple(o if isinstance(o, int) else tuple(o) for o in m["overlay"]["orders"])
        m["overlay"]["orders"] = norm
        s["orders"] = norm
        s["H"] = m["overlay"].get("H", 2)
    if m.get("hash") is not None:  # hashed cue streams: one hash head each, shared schedule
        norm = tuple(o if isinstance(o, int) else tuple(o) for o in m["hash"]["orders"])
        m["hash"]["orders"] = norm
        assert not m["overlay"] or tuple(m["overlay"]["orders"]) == norm, "hgdn + overlay must share cues"
        s["orders"], s["H"] = norm, 1
        if m["hash"].get("joint"):
            s["joint"] = True
    static_orders = m.pop("static_orders", None)
    if rows:
        m["static"] = dict(orders=s["orders"], H=s["H"], d_mem=32, rows=int(rows))
        if static_orders is not None:  # read only these cue lengths (indices into the schedule's streams)
            so = [o if isinstance(o, int) else tuple(o) for o in static_orders]
            m["static"]["streams"] = [i * s["H"] + h for i, o in enumerate(s["orders"]) if o in so
                                      for h in range(s["H"])]
            m["static"]["orders"] = tuple(so)
        s["table_rows"] = rows
    return m, s


ARMS = ["attn", "base", "base_param", "base_state", "ovl", "ovl_bounded", "ovl_multi",
        "ctrl_random", "ctrl_frozen", "ovl_only", "locker_gdn", "sdm_gdn", "ngh", "ngh123", "hgdn"]
