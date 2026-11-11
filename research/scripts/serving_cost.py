"""Serving-cost model: long-context decode for a Kimi-K3.1-shaped model with fewer sparse-attention
layers plus In-gram (codename locker, additive read) slot state in host RAM, and a Qwen3.8-Flash-Next-shaped
model on local machines.

Every number below is an ASSUMPTION, set in one place so it can be checked and changed. The model
is a roofline: decode is memory-bandwidth bound; each decode step reads every GPU's weight shard once
(large batch touches every expert) plus each session's per-token attention state. Sessions per
replica are capped by HBM left after weights. Prefill counts only the indexer's O(L^2) scoring.

In-gram state follows the frozen recipe: 6 cue lengths, and slots that grow with context. Held-out sessions
read from their first token have written 2.89 slots per token at 16k, 1.96 at 64k and 1.34 at 128k
(scripts/slots_from_start.py, results/slots_from_start.json); the scenarios assume 1.5, and the bound is one
new slot per cue length per token = 6.

    python scripts/serving_cost.py            # markdown tables to stdout
    python scripts/serving_cost.py --tex      # also write paper/generated/tab_serving_*.tex and serving_numbers.txt
"""
from __future__ import annotations

from dataclasses import dataclass, replace

GB = 1e9


@dataclass
class HW:  # per GPU / per node
    name: str = "H200 SXM"
    hbm_gb: float = 141.0
    hbm_tbs: float = 4.8  # TB/s
    fp8_tflops: float = 1979.0  # dense FP8
    usd_per_gpu_hour: float = 3.50  # assumed rental price
    gpus_per_node: int = 8
    host_ram_gb_per_node: float = 2048.0
    pcie_gbs_per_gpu: float = 64.0  # PCIe 5.0 x16, one direction
    hbm_usable: float = 0.90  # fraction left after activations / fragmentation


@dataclass
class Model:  # Kimi-K3.1-Pro-like; layer counts, indexer shape and weight precision from the model card
    # (huggingface.co/moonshotai/Kimi-K3.1-Pro, read 2026-11-10: "69 KDA + 24 Gated DSA", index 32 x 128, top-K 2048,
    # "MXFP4 weights / MXFP8 activations"); the rest assumed
    total_params_b: float = 2780.11  # parameter count of the released checkpoint (Hugging Face file listing)
    # Released checkpoint: 96 safetensors files, 1,561.3 GB = 4.49 bits per parameter on average (attention, shared
    # experts, dense MLP, router and output layer are excluded from MXFP4). Was 0.5 (4 bits flat) until the
    # 2026-11-10 accuracy audit, and 1.0 (FP8) before that.
    weight_bytes_per_param: float = 1561.293 / 2780.11
    n_linear: int = 69  # KDA layers
    n_sparse: int = 24  # gated DSA layers
    kv_bytes_tok_layer: float = 1152.0  # MLA latent 512 + 64, BF16 (assumed)
    idx_bytes_tok_layer: float = 132.0  # 128-d FP8 index key + scale (assumed)
    idx_heads: int = 32
    idx_dim: int = 128
    topk: int = 2048
    linear_state_mb_layer: float = 6.29  # KDA: 96 heads x 128 x 128 x fp32 (heads and head width from the config)
    # In-gram (additive read): one d_mem vector per written slot; slots grow with context
    locker_orders: int = 6  # cue lengths 1, 2, 4, 8, 16, 32
    locker_dmem: int = 1024  # slot width (assumed; our models use 64 at d_model 256-512)
    locker_bytes: int = 2  # BF16
    locker_slots_per_token: float = 1.5  # assumed; measured 1.34 after 131,072 tokens; bound = locker_orders
    locker_index_bytes: int = 24  # index entry per slot (8-byte key + 4-byte id at load 0.5; src/stream.py)
    locker: bool = False


def per_session(m: Model, L: int) -> dict:
    """Bytes held (GPU) and bytes read per decode step for one session at context L."""
    kv = m.n_sparse * L * m.kv_bytes_tok_layer
    idx = m.n_sparse * L * m.idx_bytes_tok_layer
    lin = m.n_linear * m.linear_state_mb_layer * 1e6
    held = kv + idx + lin
    read = idx + m.n_sparse * min(m.topk, L) * m.kv_bytes_tok_layer + 2 * lin  # indexer scan + top-k + state r/w
    host = 0.0
    host_read = 0.0
    if m.locker:
        slots = L * m.locker_slots_per_token
        host = slots * (m.locker_dmem * m.locker_bytes + m.locker_index_bytes)
        host_read = 2 * m.locker_orders * m.locker_dmem * m.locker_bytes  # one read + one write per order
    return dict(held=held, read=read, host=host, host_read=host_read, kv=kv, idx=idx, lin=lin)


def replica(hw: HW, m: Model, L: int, nodes: int, out_tokens: int = 1) -> dict:
    gpus = nodes * hw.gpus_per_node
    weights = m.total_params_b * 1e9 * m.weight_bytes_per_param
    free = gpus * hw.hbm_gb * GB * hw.hbm_usable - weights
    s = per_session(m, L)
    sessions = int(free // s["held"]) if free > 0 else 0
    if m.locker:
        sessions = min(sessions, int(nodes * hw.host_ram_gb_per_node * GB * 0.8 // max(s["host"], 1)))
    # decode step: every GPU reads its weight shard + its share of the sessions' attention reads
    step_s = (weights / gpus + sessions * s["read"] / gpus) / (hw.hbm_tbs * 1e12)
    if m.locker:  # host reads over PCIe, overlapped with HBM reads; count only if they dominate
        step_s = max(step_s, sessions * s["host_read"] / gpus / (hw.pcie_gbs_per_gpu * GB))
    tok_s = sessions / step_s if step_s else 0.0
    usd_h = gpus * hw.usd_per_gpu_hour
    usd_per_mtok = usd_h / (tok_s * 3600) * 1e6 if tok_s else float("inf")
    prefill_flops = m.n_sparse * L * L * m.idx_heads * m.idx_dim * 2 / 2  # causal half
    prefill_s = prefill_flops / (gpus * hw.fp8_tflops * 1e12 * 0.5)  # 50% utilization assumed
    return dict(gpus=gpus, sessions=sessions, step_ms=step_s * 1e3, tok_s=tok_s, usd_per_mtok=usd_per_mtok,
                kv_gb=s["kv"] / GB, idx_gb=s["idx"] / GB, held_gb=s["held"] / GB, read_gb=s["read"] / GB,
                host_gb=s["host"] / GB, prefill_s=prefill_s)


def table(hw: HW, base: Model, L: int, nodes: int, scenarios) -> str:
    rows = ["| config | sparse layers | GPU state / session (GB) | host RAM / session (GB) | attn read / token / session (GB) "
            "| sessions / replica | decode step (ms) | tokens/s / replica | $ / M output tokens | indexer prefill (s) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in scenarios:
        r = replica(hw, m, L, nodes)
        rows.append(f"| {name} | {m.n_sparse} | {r['held_gb']:.1f} | {r['host_gb']:.1f} | {r['read_gb']:.2f} | "
                    f"{r['sessions']} | {r['step_ms']:.1f} | {r['tok_s']:.0f} | {r['usd_per_mtok']:.2f} | {r['prefill_s']:.1f} |")
    return "\n".join(rows)


def main():
    hw = HW()
    base = Model()
    nodes = 4  # 32 x H200
    for L in (131072, 1048576):
        sc = [("today (24 DSA)", base),
              ("12 DSA + In-gram", replace(base, n_sparse=12, n_linear=81, locker=True)),
              ("6 DSA + In-gram", replace(base, n_sparse=6, n_linear=87, locker=True))]
        print(f"\n### Context {L // 1024}k, one replica = {nodes} nodes x {hw.gpus_per_node} {hw.name} "
              f"(${hw.usd_per_gpu_hour}/GPU-h assumed)\n")
        print(table(hw, base, L, nodes, sc))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# Local rigs: one user, batch 1. Decode is bandwidth bound: each token reads the active weights,
# every attention layer's whole KV cache, and the linear layers' state; prefill is compute bound.

@dataclass
class Rig:
    name: str
    mem_gb: float  # memory the model may use (GPU VRAM, or Mac unified memory after the OS share)
    bw_gbs: float  # bandwidth of that memory
    tflops: float  # FP16 throughput usable for prefill
    offload_gb: float = 0.0  # system RAM for offloaded MoE experts (PC with a discrete GPU)
    offload_bw_gbs: float = 0.0


@dataclass
class LocalModel:  # Qwen3.8-Flash-Next, from its config.json on Hugging Face (2026-11-08) and tech report
    total_b: float = 125.0  # backbone params (MoE: 512 experts, 10 active)
    active_b: float = 6.0
    expert_frac: float = 0.93  # share of backbone weights in routed experts (offloadable); estimate
    bits: float = 4.5  # Q4_K-style weights incl. scales (community quantization; assumed)
    ngram_b: float = 51.0  # n-gram embedding tables (host memory in Qwen's own serving)
    ngram_bits: float = 4.5  # assumed quantized like the weights for local use
    n_layers: int = 48
    n_attn: int = 12  # full_attention_interval 4 -> 12 QSA layers, 36 GDN
    kv_heads: int = 2
    head_dim: int = 256
    q_heads: int = 24
    kv_bytes: float = 2.0  # FP16 cache
    topk: int = 2048  # QSA token budget per query
    idx_bytes_tok: float = 64.0  # QSA indexer key: 128 dims FP16, compressed 4 tokens -> 1 block
    gdn_state_mb: float = 3.1  # 48 value heads x 128 x 128 x fp32 per GDN layer, read + written per token
    locker: bool = False
    locker_orders: int = 6
    locker_dmem: int = 512  # slot width (assumed)
    locker_bytes: float = 2.0  # FP16
    locker_slots_per_token: float = 1.5  # assumed; measured 1.34 after 131,072 tokens; bound = locker_orders
    locker_index_bytes: int = 24
    locker_table_rows: int = 1 << 16  # static table rows per cue length, as in our recipe (assumed to carry over)


def local(rig: Rig, m: LocalModel, L: int) -> dict:
    w = m.total_b * 1e9 * m.bits / 8
    ng = m.ngram_b * 1e9 * m.ngram_bits / 8
    kv_tok = m.n_attn * 2 * m.kv_heads * m.head_dim * m.kv_bytes
    idx_tok = m.n_attn * m.idx_bytes_tok
    # In-gram: a static table (parameters) plus per-session slot state that grows with context
    lock_tab = m.locker_orders * m.locker_table_rows * m.locker_dmem * m.locker_bytes if m.locker else 0.0
    lock_tok = m.locker_slots_per_token * (m.locker_dmem * m.locker_bytes + m.locker_index_bytes) if m.locker else 0.0
    if rig.offload_gb:  # experts + n-gram tables + In-gram table and slots in system RAM; the rest in VRAM
        in_vram = w * (1 - m.expert_frac)
        free = rig.mem_gb * GB - in_vram
        max_ctx = int(free / (kv_tok + idx_tok)) if free > 0 else 0
        ram_free = rig.offload_gb * GB - w * m.expert_frac - ng - lock_tab
        fits_ram = ram_free >= 0
        if m.locker:
            max_ctx = min(max_ctx, int(max(ram_free, 0) / lock_tok))
    else:  # unified memory holds everything
        free = rig.mem_gb * GB - w - ng - lock_tab
        fits_ram = True
        max_ctx = int(free / (kv_tok + idx_tok + lock_tok)) if free > 0 else 0
    act = m.active_b * 1e9 * m.bits / 8
    lin = (m.n_layers - m.n_attn) * 2 * m.gdn_state_mb * 1e6
    attn_read = m.n_attn * min(L, m.topk) * 2 * m.kv_heads * m.head_dim * m.kv_bytes + L * idx_tok
    if rig.offload_gb:
        t = max(act * m.expert_frac / (rig.offload_bw_gbs * GB),
                (act * (1 - m.expert_frac) + attn_read + lin) / (rig.bw_gbs * GB))
    else:
        t = (act + attn_read + lin) / (rig.bw_gbs * GB)
    # prefill: QSA indexer O(L^2 / r) + sparse attention O(L K) per attention layer, plus the MoE
    idx_flops = m.n_attn * 4 * 128 * L * (L / 4) / 2 * 2
    att_flops = m.n_attn * 4 * m.q_heads * m.head_dim * L * min(L, m.topk)
    prefill = (idx_flops + att_flops + 2 * m.active_b * 1e9 * L) / (rig.tflops * 1e12 * 0.4)
    return dict(max_ctx=max_ctx, fits=(L <= max_ctx) and fits_ram, tok_s=1 / t, prefill_min=prefill / 60,
                kv_gb=L * (kv_tok + idx_tok) / GB, fits_ram=fits_ram, kv_tok=kv_tok + idx_tok, lock_tok=lock_tok,
                lock_gb=L * lock_tok / GB, lock_tab_gb=lock_tab / GB)


RIGS = [Rig("RTX 4090 24 GB + 128 GB DDR5", 22, 1008, 165, offload_gb=120, offload_bw_gbs=80),
        Rig("MacBook Pro, 128 GB (M4 Max-class)", 110, 546, 34),
        Rig("DGX Spark, 128 GB (GB10, 273 GB/s)", 115, 273, 100)]  # FP16 dense TFLOPS assumed
LOCAL_VARIANTS = [("today 3:1 (12 attn)", LocalModel()),
                  ("In-gram, 6 attn", replace(LocalModel(), n_attn=6, locker=True)),
                  ("In-gram, 4 attn", replace(LocalModel(), n_attn=4, locker=True))]


def local_tables():
    rigs = RIGS
    variants = LOCAL_VARIANTS
    out = []
    for rig in rigs:
        out.append(f"\n### {rig.name}\n")
        out.append("| model | max context that fits | KV + index at 256k (GB) | decode tok/s @128k | @256k | @1M | prefill 128k (min) |")
        out.append("|---|---|---|---|---|---|---|")
        for name, m in variants:
            r = {L: local(rig, m, L) for L in (131072, 262144, 1048576)}
            cell = lambda L: f"{r[L]['tok_s']:.0f}" if r[L]["fits"] else "doesn't fit"
            out.append(f"| {name} | {r[131072]['max_ctx'] // 1024}k | {r[262144]['kv_gb']:.1f} | {cell(131072)} | "
                       f"{cell(262144)} | {cell(1048576)} | {r[131072]['prefill_min']:.1f} |")
    return "\n".join(out)


if __name__ == "__main__":
    print("\n## Local rigs (batch 1)")
    print(local_tables())


# ---------------------------------------------------------------------------------------------
# LaTeX tables and prose numbers for paper/sections/serving.tex

def write_tex():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]  # paper/ sits beside scripts/, or one level up in the public release
    out = (root if (root / "paper").exists() else root.parent) / "paper" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    hdr = "% Generated by scripts/serving_cost.py --tex (a roofline model; every input is an assumption). Do not edit.\n"
    notes = []
    k = lambda n: f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"

    # local machines: largest context that fits, by attention layers kept
    rows = []
    for rig, label in zip(RIGS, ("RTX 4090, 24 GB + 128 GB RAM", "Laptop, 128 GB unified memory", "Mini PC, 128 GB unified memory")):
        r = [local(rig, m, 262144) for _, m in LOCAL_VARIANTS]
        rows.append(f"{label} & {k(r[0]['max_ctx'])} & {k(r[1]['max_ctx'])} & {k(r[2]['max_ctx'])} \\\\")
        notes.append(f"{rig.name}: max context {[x['max_ctx'] for x in r]}; decode tok/s at 256k "
                     f"{[round(x['tok_s']) for x in r]}; fits 256k {[x['fits'] for x in r]}")
    r = [local(RIGS[1], m, 1048576) for _, m in LOCAL_VARIANTS]
    notes.append(f"per token (bytes): KV + index {[round(x['kv_tok']) for x in r]}, In-gram slots {[round(x['lock_tok']) for x in r]}; "
                 f"In-gram static table {r[1]['lock_tab_gb']:.2f} GB; at 1M tokens: KV + index {[round(x['kv_gb'], 1) for x in r]} GB, "
                 f"In-gram slots {[round(x['lock_gb'], 1) for x in r]} GB")
    bound = [local(RIGS[1], replace(m, locker_slots_per_token=6.0), 262144)["max_ctx"] for _, m in LOCAL_VARIANTS]
    notes.append(f"laptop, at the bound of 6 slots per token: max context {bound}")
    (out / "tab_serving_local.tex").write_text(hdr + "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "& 12 attention layers & 6 attention layers & 4 attention layers \\\\\n"
        "Machine & (as released) & + \\ingram{} & + \\ingram{} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # datacenter replica: sessions that fit, by sparse-attention layers kept
    hw, base, nodes = HW(), Model(), 4
    sc = [("24 attention layers (as released)", base),
          ("12 attention layers + \\ingram{}", replace(base, n_sparse=12, n_linear=81, locker=True)),
          ("6 attention layers + \\ingram{}", replace(base, n_sparse=6, n_linear=87, locker=True))]
    rows = []
    for name, m in sc:
        a, b = replica(hw, m, 131072, nodes), replica(hw, m, 1048576, nodes)
        rows.append(f"{name} & {a['held_gb']:.1f} & {a['host_gb']:.1f} & {a['sessions']:,} & {b['held_gb']:.1f} & {b['host_gb']:.1f} & {b['sessions']:,} \\\\")
        notes.append(f"{name}: 128k held {a['held_gb']:.2f} GB host {a['host_gb']:.2f} GB sessions {a['sessions']} step {a['step_ms']:.1f} ms "
                     f"tok/s {a['tok_s']:.0f}; 1M held {b['held_gb']:.2f} GB host {b['host_gb']:.2f} GB sessions {b['sessions']} "
                     f"step {b['step_ms']:.1f} ms tok/s {b['tok_s']:.0f}")
    (out / "tab_serving_dc.tex").write_text(hdr + "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        "& \\multicolumn{3}{c}{128k-token sessions} & \\multicolumn{3}{c}{1M-token sessions} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "& GPU state & host state & sessions & GPU state & host state & sessions \\\\\n"
        "Configuration & (GB) & (GB) & per replica & (GB) & (GB) & per replica \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "serving_numbers.txt").write_text("\n".join(notes) + "\n")
    print("\n".join(notes))


if __name__ == "__main__":
    import sys

    if "--tex" in sys.argv:
        print("\n## Numbers for paper/sections/serving.tex")
        write_tex()
