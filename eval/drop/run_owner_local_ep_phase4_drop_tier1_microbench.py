"""Tier 1 microbenchmark for the Prefill L Sweep.

Sweeps (T_local, drop_rate) in random order on an isolated 1-layer EP-HT MoE
block constructed against the production DispatchEPHT/ExpertsEPHT/CombineEPHT
artifacts. Times the three segments (dispatch, experts, combine) plus the
total with CUDA events; aggregates across ranks (rank-max per repeat) and
writes one JSONL row per (cell, repeat) plus a summary.json with the L*
decision and pre-registration check.

Usage (8-rank main sweep):

    torchrun --nproc_per_node=8 --master_port=29500 \\
        -m eval.drop.tier1_bench \\
        --output-dir eval_results/prefill_drop_l_sweep_tier1 \\
        --t-local-values 1,8,64,512,2048 \\
        --drop-rates 0.0,0.3 \\
        --warmup-iters 10 --iters 30 \\
        --expert-overlap-path eval_results/.../moe_overlap_plan_smoke...json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from types import SimpleNamespace

# Drop env vars must be set BEFORE importing dispatch_ep_ht (it reads them at
# __init__ time). Use setdefault so callers can still override on the CLI side.
os.environ.setdefault("MOE_DROP_IMPL", "gpu")
os.environ.setdefault("MOE_DROP_MIN_REPLICAS", "0")
os.environ.setdefault("MOE_DROP_GPU_STATS", "0")
os.environ.setdefault("MOE_PROFILE_OVERLAP_RUNTIME", "0")
os.environ.setdefault("MOE_RECORD_TIMING", "0")
os.environ.setdefault("MOE_PROFILE_ROUTING", "0")

import torch
import torch.distributed as dist

from eval.drop.shared import (
    CUDAEventTimer,
    WandbLogger,
    agg_stats,
    truncate_file,
    write_jsonl,
)
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.dispatch_ep_ht import DispatchEPHT
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.experts_ep_ht import ExpertsEPHT
from workshop.nanovllm_moe.artifacts.modeling.layers.moe.combine_ep_ht import CombineEPHT
from workshop.nanovllm_moe.artifacts.moe_backend.moe_backend import MoeBackend
from workshop.nanovllm_moe.services.utils.parallel import init_parallel_groups


# ---------------------------------------------------------------------------
# Model shape (Qwen3-30B-A3B)
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 768
NUM_EXPERTS_GLOBAL = 128
TOP_K = 8
BLOCK_M = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier 1 Prefill L Sweep microbenchmark")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--t-local-values", default="1,8,64,512,2048",
                   help="Comma-separated T_local values; L_send_nominal = T*K")
    p.add_argument("--drop-rates", default="0.0,0.3",
                   help="Comma-separated drop_rate values")
    p.add_argument("--drop-policy", default="tail_weight")
    p.add_argument("--drop-seed", type=int, default=42)
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--expert-overlap-path", default="",
                   help="LBG overlap plan JSON. Empty -> disjoint placement.")
    p.add_argument("--expert-overlap-strategy", default="greedy_balance")
    p.add_argument("--expert-placement", default="contiguous")
    p.add_argument("--cooldown-sec", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="",
                   help="W&B project name. Empty -> wandb disabled.")
    p.add_argument("--wandb-name", default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    torch.set_default_device(f"cuda:{local_rank}")
    init_parallel_groups(
        tp_size=1,
        world_size=world,
        data_parallel_size=world,
        runtime_mode="owner_local_ep",
    )
    return rank, local_rank, world


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_moe_block(args: argparse.Namespace, world: int, max_T_local: int):
    """Construct 1-layer EP-HT MoE block (backend + dispatch + experts + combine)."""
    overlap_enabled = bool(args.expert_overlap_path)
    overlap_path = args.expert_overlap_path or None

    config = SimpleNamespace(
        max_num_batched_tokens=max_T_local,
        moe_block_size_m=BLOCK_M,
        moe_impl="ep_ht",
        enforce_eager=True,
        hf_config=SimpleNamespace(torch_dtype=torch.bfloat16),
        moe_expert_overlap_enabled=overlap_enabled,
        moe_expert_overlap_path=overlap_path,
    )
    backend = MoeBackend(
        config,
        num_experts=NUM_EXPERTS_GLOBAL,
        top_k=TOP_K,
        hidden_size=HIDDEN_SIZE,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
    )

    dispatch = DispatchEPHT(
        num_experts_global=NUM_EXPERTS_GLOBAL,
        top_k=TOP_K,
        block_size_m=BLOCK_M,
        norm_topk_prob=True,
        expert_placement=args.expert_placement,
        expert_overlap_enabled=overlap_enabled,
        expert_overlap_path=overlap_path,
        expert_overlap_strategy=args.expert_overlap_strategy,
        drop_policy=args.drop_policy,
        drop_rate=0.0,
        drop_seed=args.drop_seed,
        router_keff=0,
        layer_id=0,
    ).cuda()
    # Attach backend buffers so the alignment / cumsum workspaces don't get
    # resized cell-by-cell.
    dispatch.sorted_token_ids_buf = backend.sorted_token_ids_buf
    dispatch.expert_ids_buf = backend.expert_ids_buf
    dispatch.num_tokens_post_padded = backend.num_tokens_post_padded
    dispatch.cumsum_buffer = backend.cumsum_buffer

    experts = ExpertsEPHT(
        num_experts_global=NUM_EXPERTS_GLOBAL,
        hidden_size=HIDDEN_SIZE,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
        expert_placement=args.expert_placement,
        expert_overlap_enabled=overlap_enabled,
        expert_overlap_path=overlap_path,
        layer_id=0,
    ).cuda()
    experts.run_experts = backend.run_experts
    with torch.no_grad():
        # ExpertsEPHT allocates w1/w2 with default dtype (fp32); cast to bf16 to
        # match the kernel's bf16 GEMM contract.
        experts.w1.data = experts.w1.data.to(torch.bfloat16)
        experts.w2.data = experts.w2.data.to(torch.bfloat16)
        experts.w1.data.normal_(mean=0.0, std=0.02)
        experts.w2.data.normal_(mean=0.0, std=0.02)

    combine = CombineEPHT(hidden_size=HIDDEN_SIZE, top_k=TOP_K).cuda()
    return backend, dispatch, experts, combine


def prebuild_inputs(T_locals: list[int], seed: int, device: str) -> dict:
    inputs = {}
    for T in T_locals:
        gen_h = torch.Generator(device=device).manual_seed(seed * 100003 + T)
        h = torch.randn(
            (T, HIDDEN_SIZE), dtype=torch.bfloat16, generator=gen_h, device=device,
        )
        gen_l = torch.Generator(device=device).manual_seed(seed * 7919 + T)
        l = torch.randn(
            (T, NUM_EXPERTS_GLOBAL), dtype=torch.float32, generator=gen_l, device=device,
        )
        inputs[T] = (h, l)
    return inputs


# ---------------------------------------------------------------------------
# Per-cell timing
# ---------------------------------------------------------------------------

def run_one_cell(
    cell_id: int,
    T_local: int,
    drop_rate: float,
    backend,
    dispatch,
    experts,
    combine,
    hidden,
    logits,
    world: int,
    args: argparse.Namespace,
) -> list[dict]:
    # Mutate drop rate in-place (avoid module rebuild).
    dispatch.drop_rate = drop_rate
    dispatch.drop_enabled = (
        dispatch.drop_policy != "none" and drop_rate > 0.0
    )
    backend.prepare_metadata_for_moe(T_local)

    # Warmup.
    dist.barrier()
    for _ in range(args.warmup_iters):
        tok_meta = dispatch(hidden, logits)
        eo = experts(tok_meta)
        _ = combine(eo, tok_meta)
    torch.cuda.synchronize()

    rows: list[dict] = []
    for rep in range(args.iters):
        d_t = CUDAEventTimer()
        e_t = CUDAEventTimer()
        c_t = CUDAEventTimer()
        tot_t = CUDAEventTimer()

        dist.barrier()
        with tot_t:
            with d_t:
                tok_meta = dispatch(hidden, logits)
            with e_t:
                eo = experts(tok_meta)
            with c_t:
                _ = combine(eo, tok_meta)
        torch.cuda.synchronize()

        d_us = d_t.read()
        e_us = e_t.read()
        c_us = c_t.read()
        t_us = tot_t.read()
        L_recv = int(tok_meta.recv_hidden.shape[0])
        L_send_kept = int(sum(tok_meta.send_counts))

        # Cross-rank aggregation: each segment uses rank-max (slowest decides).
        local = torch.tensor(
            [d_us, e_us, c_us, t_us, float(L_recv), float(L_send_kept)],
            dtype=torch.float64, device="cuda",
        )
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        mat = torch.stack(gathered).cpu().numpy()
        rows.append({
            "cell_id": cell_id,
            "T_local": T_local,
            "drop_rate": drop_rate,
            "repeat_id": rep,
            "dispatch_us_rank_max": float(mat[:, 0].max()),
            "experts_us_rank_max": float(mat[:, 1].max()),
            "combine_us_rank_max": float(mat[:, 2].max()),
            "total_us_rank_max": float(mat[:, 3].max()),
            "L_recv_mean": float(mat[:, 4].mean()),
            "L_recv_max": float(mat[:, 4].max()),
            "L_recv_min": float(mat[:, 4].min()),
            "L_send_kept_mean": float(mat[:, 5].mean()),
            "L_send_nominal": T_local * TOP_K,
        })

    return rows


# ---------------------------------------------------------------------------
# Post-process: L* and decision
# ---------------------------------------------------------------------------

ALPHA = -0.05  # main delta_pct threshold (-5%)
# Diagnostic only: large-L drop should be at least as fast as baseline if drop
# kernel really is zero-overhead. If at the LARGEST L we still see significant
# slowdown, the drop kernel itself is suspect (not a workload issue).
LARGE_L_REGRESSION = 0.10  # +10% at max L = regression flag


def compute_l_star(delta_points: list[dict]) -> tuple[float | None, str]:
    """delta_points sorted by L_recv_max_p50 asc, each has `delta_pct`."""
    if not delta_points:
        return None, "no_data"
    if delta_points[0]["delta_pct"] <= ALPHA:
        return delta_points[0]["L_recv_max_p50"], "crossing_below_min"
    for i in range(len(delta_points) - 1):
        p0 = delta_points[i]["delta_pct"]
        p1 = delta_points[i + 1]["delta_pct"]
        L0 = delta_points[i]["L_recv_max_p50"]
        L1 = delta_points[i + 1]["L_recv_max_p50"]
        if p0 > ALPHA and p1 <= ALPHA:
            L_star = L0 + (ALPHA - p0) * (L1 - L0) / (p1 - p0)
            return float(L_star), "crossing_found"
    return None, "no_crossing"


def write_summary(jsonl_path: str, output_dir: str) -> None:
    from collections import defaultdict

    with open(jsonl_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        print("[summary] no rows")
        return

    cells = defaultdict(list)
    for r in rows:
        cells[(r["T_local"], r["drop_rate"])].append(r)

    cell_agg: dict[tuple, dict] = {}
    for (T, rate), rs in cells.items():
        cell_agg[(T, rate)] = {
            "T_local": T,
            "drop_rate": rate,
            "iters": len(rs),
            "total_us": agg_stats([r["total_us_rank_max"] for r in rs]),
            "dispatch_us": agg_stats([r["dispatch_us_rank_max"] for r in rs]),
            "experts_us": agg_stats([r["experts_us_rank_max"] for r in rs]),
            "combine_us": agg_stats([r["combine_us_rank_max"] for r in rs]),
            "L_recv_max": agg_stats([r["L_recv_max"] for r in rs]),
            "L_recv_mean": agg_stats([r["L_recv_mean"] for r in rs]),
            "L_send_kept_mean": agg_stats([r["L_send_kept_mean"] for r in rs]),
        }

    rates = sorted({rate for _, rate in cell_agg.keys()})
    baseline_rate = min(rates)               # usually 0.0
    drop_rate_main = max(rates) if len(rates) > 1 else baseline_rate

    T_locals = sorted({T for T, _ in cell_agg.keys()})
    delta_points: list[dict] = []
    for T in T_locals:
        base = cell_agg.get((T, baseline_rate))
        drop = cell_agg.get((T, drop_rate_main))
        if base is None or drop is None or base["total_us"]["mean"] <= 0:
            continue
        L = base["L_recv_max"]["p50"]
        delta = drop["total_us"]["mean"] - base["total_us"]["mean"]
        delta_pct = delta / base["total_us"]["mean"]
        denom = base["total_us"]["mean"] - drop["total_us"]["mean"]
        ratio_experts = (
            (base["experts_us"]["mean"] - drop["experts_us"]["mean"]) / denom
            if abs(denom) > 1e-6 else None
        )
        eff_send = 1.0 - (drop["L_send_kept_mean"]["mean"] / max(T * TOP_K, 1))
        eff_recv = (
            1.0 - drop["L_recv_mean"]["mean"] / base["L_recv_mean"]["mean"]
            if base["L_recv_mean"]["mean"] > 0 else 0.0
        )
        delta_points.append({
            "T_local": T,
            "L_send_nominal": T * TOP_K,
            "L_recv_max_p50": L,
            "L_recv_mean_p50": base["L_recv_mean"]["p50"],
            "baseline_total_us_mean": base["total_us"]["mean"],
            "drop_total_us_mean": drop["total_us"]["mean"],
            "delta_us": delta,
            "delta_pct": delta_pct,
            "ratio_experts": ratio_experts,
            "effective_drop_send_frac": eff_send,
            "effective_drop_recv_frac": eff_recv,
        })
    delta_points.sort(key=lambda d: d["L_recv_max_p50"])

    L_star, status = compute_l_star(delta_points)
    h1_refuted = status in ("crossing_found", "crossing_below_min")

    h2_ratio = None
    if h1_refuted and L_star is not None:
        anchor = min(
            delta_points,
            key=lambda d: abs(d["L_recv_max_p50"] - L_star),
        )
        h2_ratio = anchor["ratio_experts"]

    # Diagnostic: large-L should not regress if drop is truly zero-overhead.
    large_dp = delta_points[-1] if delta_points else None
    large_L_regressed = (
        large_dp is not None and large_dp["delta_pct"] > LARGE_L_REGRESSION
    )

    if h1_refuted:
        decision = "RUN_TIER2"
    else:
        decision = "STOP_CLOSED_NEGATIVE"

    summary = {
        "alpha": ALPHA,
        "baseline_rate": baseline_rate,
        "drop_rate_main": drop_rate_main,
        "L_star": L_star,
        "L_star_status": status,
        "h1_status": "refuted" if h1_refuted else "not_refuted",
        "h2_ratio_experts": h2_ratio,
        "decision": decision,
        "pre_reg_check": {
            "small_L_drop_overhead_pct": delta_points[0]["delta_pct"] if delta_points else None,
            "large_L_drop_overhead_pct": large_dp["delta_pct"] if large_dp else None,
            "large_L_regression_threshold": LARGE_L_REGRESSION,
            "large_L_regressed": large_L_regressed,
        },
        "delta_points": delta_points,
        "cell_agg": {
            f"T{T}_r{rate}": v for (T, rate), v in cell_agg.items()
        },
    }

    out_path = os.path.join(output_dir, "tier1_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] decision={decision}  L_star={L_star}  h1={summary['h1_status']}")
    print(f"[summary] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    rank, local_rank, world = setup_distributed()
    seed_all(args.seed)
    device = f"cuda:{local_rank}"

    T_locals = sorted({int(x) for x in args.t_local_values.split(",")})
    rates = sorted({float(x) for x in args.drop_rates.split(",")})

    if rank == 0:
        print(f"[bench] world={world} T_locals={T_locals} rates={rates} "
              f"warmup={args.warmup_iters} iters={args.iters}")

    max_T = max(T_locals)
    backend, dispatch, experts, combine = build_moe_block(args, world, max_T)

    inputs_cache = prebuild_inputs(T_locals, args.seed, device)

    cells = [(T, r) for T in T_locals for r in rates]
    rng = random.Random(args.seed)   # same seed across ranks -> identical order
    rng.shuffle(cells)

    jsonl_path = os.path.join(args.output_dir, "tier1_rows.jsonl")
    truncate_file(jsonl_path)
    dist.barrier()

    wandb_name = args.wandb_name or f"tier1_w{world}_seed{args.seed}"
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=wandb_name,
        config=vars(args),
        enabled=bool(args.wandb_project),
    )

    for ci, (T_local, drop_rate) in enumerate(cells):
        if rank == 0:
            print(f"[cell {ci+1}/{len(cells)}] T_local={T_local} drop_rate={drop_rate}")
        hidden, logits = inputs_cache[T_local]
        try:
            rows = run_one_cell(
                cell_id=ci, T_local=T_local, drop_rate=drop_rate,
                backend=backend, dispatch=dispatch, experts=experts, combine=combine,
                hidden=hidden, logits=logits, world=world, args=args,
            )
        except RuntimeError as exc:
            if rank == 0:
                print(f"[cell {ci+1}] RuntimeError: {exc}")
            dist.barrier()
            continue
        if rank == 0:
            for r in rows:
                write_jsonl(jsonl_path, r)
                wandb_logger.log(r)
        torch.cuda.synchronize()
        dist.barrier()
        if ci != len(cells) - 1 and args.cooldown_sec > 0:
            time.sleep(args.cooldown_sec)

    dist.barrier()
    if rank == 0:
        write_summary(jsonl_path, args.output_dir)
    wandb_logger.finish()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
