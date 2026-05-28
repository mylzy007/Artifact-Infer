# Prefill L Sweep — Task-Level Plan（TDD）

> **For agentic workers:** plan → tests → impl，每步可独立 commit。
>
> **Goal**：实现 `eval/drop/` 下 standalone Tier 1 + Tier 2 microbench，48h 内完成 H1 判定。
>
> **Architecture**：3 个独立 module
> - `eval/drop/shared.py` —— CUDA event timer / JSONL / W&B / agg
> - `eval/drop/tier1_bench.py` —— python -m runnable，torchrun 8 ranks，sweep `(T_local, drop_rate)`
> - `eval/drop/tier2_bench.py` —— python -m runnable，仅在 H1 推翻时跑
>
> **Tech**：PyTorch + torch.distributed (NCCL) + CUDA events + 直接复用 `DispatchEPHT`/`ExpertsEPHT`/`CombineEPHT`/`MoeBackend`。

---

## 关键接口契约（来自代码 explore）

| 组件 | 文件:行 | 关键签名 |
|---|---|---|
| `DispatchEPHT.__init__` | `workshop/nanovllm_moe/artifacts/modeling/layers/moe/dispatch_ep_ht.py:95` | `(num_experts_global, top_k, block_size_m, norm_topk_prob, expert_placement, expert_placement_seed, expert_placement_path, expert_overlap_enabled, expert_overlap_path, expert_overlap_strategy, drop_policy, drop_rate, drop_seed, router_keff, layer_id)` |
| `DispatchEPHT.forward` | `dispatch_ep_ht.py:228` | `(hidden_states [T,H] bf16, router_logits [T,E] fp32) -> TokMetaEPHT` |
| `TokMetaEPHT` | `dispatch_ep_ht.py:66` | NamedTuple，需用字段：`recv_hidden`, `send_counts`, `recv_counts`, `topk_weights`, `topk_ids`, `sort_perm`, `T_local` |
| `ExpertsEPHT.__init__` | `experts_ep_ht.py:34` | `(num_experts_global, hidden_size, moe_intermediate_size, expert_placement, expert_placement_seed, expert_placement_path, expert_overlap_enabled, expert_overlap_path, layer_id)`；weights `w1 [E_local, 2N, H]`, `w2 [E_local, H, N]` bf16 |
| `ExpertsEPHT.forward` | `experts_ep_ht.py:116` | `(tok_meta) -> [total_recv, 1, H]`；需手动 `experts.run_experts = backend.run_experts` |
| `CombineEPHT` | `combine_ep_ht.py:39` | `(hidden_size, top_k)`；forward `(expert_out, tok_meta) -> [T, H]` |
| `MoeBackend` | `moe_backend/moe_backend.py:30` | `(config, num_experts, top_k, hidden_size, moe_intermediate_size)`；从 `config.max_num_batched_tokens` 算 `T_cap` |
| `init_parallel_groups` | `services/utils/parallel.py:19` | `(tp_size=1, world_size=W, *, data_parallel_size=W, runtime_mode="owner_local_ep")` |
| `apply_drop_gpu_simple` | `services/utils/expert_drop.py:441` | 直接可调，返回 `DropResult.keep_mask`；适合单卡 unit test |
| 现成 LBG plan | `eval_results/tmp_phase3_overlap_test/moe_overlap_plan_smoke_plan_20260521_222623.json` | `hosts_by_expert: expert_id -> [rank_list]` |

**Env vars**（脚本内 `os.environ.setdefault`）：

```
MOE_DROP_IMPL=gpu
MOE_DROP_MIN_REPLICAS=0
MOE_DROP_GPU_STATS=0
MOE_PROFILE_OVERLAP_RUNTIME=0
MOE_RECORD_TIMING=0
```

---

## 文件结构

```
eval/drop/
  __init__.py
  shared.py
  tier1_bench.py
  tier2_bench.py
  tests/
    __init__.py
    test_drop_invariants.py
```

---

## Task 1 — `eval/drop/shared.py`

**Files**: Create `eval/drop/__init__.py`（空）, Create `eval/drop/shared.py`

**职责**：4 个工具，互不依赖，单卡可 import。

- [ ] **Step 1.1** Create `eval/drop/__init__.py`（空文件）

- [ ] **Step 1.2** 写 `CUDAEventTimer` context manager

```python
class CUDAEventTimer:
    """记录单段 CUDA 时间（μs）。用法：
        with CUDAEventTimer() as t:
            ...
        elapsed_us = t.elapsed_us  # 仅在 cuda.synchronize() 后可读
    """
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.elapsed_us: float | None = None

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        return False

    def read(self) -> float:
        """调用前必须 torch.cuda.synchronize()。"""
        self.elapsed_us = self.start.elapsed_time(self.end) * 1000.0
        return self.elapsed_us
```

- [ ] **Step 1.3** 写 `write_jsonl(path: str, row: dict)` —— append 模式 + rank0 guard

```python
def write_jsonl(path: str, row: dict) -> None:
    """rank0-only append。其他 rank no-op。"""
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
```

- [ ] **Step 1.4** 写 `agg_stats(values: list[float]) -> dict` —— mean/std/p10/p50/p90/p99

```python
def agg_stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
```

- [ ] **Step 1.5** 写 `WandbLogger` —— optional，detect via `WANDB_API_KEY`/`wandb.api.api_key`

```python
class WandbLogger:
    """No-op 如果 wandb 未安装或未配置。rank0-only。"""
    def __init__(self, project: str, name: str, config: dict, enabled: bool = True):
        self.run = None
        if not enabled:
            return
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        try:
            import wandb
            if not os.environ.get("WANDB_API_KEY") and not getattr(wandb.api, "api_key", None):
                print("[wandb] no API key configured; disabling")
                return
            self.run = wandb.init(project=project, name=name, config=config, reinit=True)
        except Exception as exc:
            print(f"[wandb] init failed: {exc}; disabling")
            self.run = None

    def log(self, row: dict) -> None:
        if self.run is None:
            return
        self.run.log(row)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
```

- [ ] **Step 1.6** Commit

---

## Task 2 — Unit tests `eval/drop/tests/test_drop_invariants.py`

**Files**: Create `eval/drop/tests/__init__.py`, Create `eval/drop/tests/test_drop_invariants.py`

**TDD 原则**：先写测试，跑一次确认能 import + 调到 `apply_drop_gpu_simple`，然后才动 bench。

**约束**：单 GPU，不需要 torchrun；直接 `python -m eval.drop.tests.test_drop_invariants`。

- [ ] **Step 2.1** Create `eval/drop/tests/__init__.py`（空文件）

- [ ] **Step 2.2** 写 `test_drop_rate_zero_keeps_all`：

```python
def test_drop_rate_zero_keeps_all():
    """drop_rate=0 时 keep_mask 全 1（数值精度内无丢失）。"""
    device = torch.device("cuda:0")
    T, K, world = 64, 8, 8
    source = 0
    g = torch.Generator(device="cpu").manual_seed(42)
    eid = torch.randint(0, world * 16, (T * K,), generator=g, dtype=torch.int32, device=device)
    tr = (eid.long() % world).to(torch.int64).to(device)
    w = torch.softmax(torch.randn(T, K, generator=g), dim=-1).view(-1).contiguous().to(device)
    gen = torch.Generator(device=device).manual_seed(7)
    res = apply_drop_gpu_simple(
        flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
        source_rank=source, world_size=world, K=K,
        drop_policy="tail_weight", drop_rate=0.0,
        torch_generator=gen, collect_stats=False,
    )
    assert res.keep_mask.all().item(), "drop_rate=0 should keep everything"
    print("OK test_drop_rate_zero_keeps_all")
```

- [ ] **Step 2.3** 写 `test_effective_drop_frac_matches`：

```python
def test_effective_drop_frac_matches():
    """大 T 下 effective_drop_send_frac ≈ drop_rate（误差 ≤ 0.05），
    扣除 local-preserve 与 per-token min-keep=1 的约束。"""
    device = torch.device("cuda:0")
    T, K, world = 4096, 8, 8
    source = 0
    g = torch.Generator(device="cpu").manual_seed(123)
    eid = torch.randint(0, world * 16, (T * K,), generator=g, dtype=torch.int32, device=device)
    # 强制 routing 远端，避免 local-preserve 主导 effective frac
    tr = ((eid.long() % (world - 1)) + 1).to(torch.int64).to(device)
    tr = (tr + source) % world  # source 永远不被选中
    w = torch.softmax(torch.randn(T, K, generator=g), dim=-1).view(-1).contiguous().to(device)
    gen = torch.Generator(device=device).manual_seed(7)
    for rate in (0.1, 0.3, 0.5):
        res = apply_drop_gpu_simple(
            flat_expert_ids=eid, target_rank=tr, flat_topk_w=w,
            source_rank=source, world_size=world, K=K,
            drop_policy="tail_weight", drop_rate=rate,
            torch_generator=gen, collect_stats=False,
        )
        kept = int(res.keep_mask.sum().item())
        effective_dropped = 1.0 - kept / (T * K)
        # 允许 5% 绝对误差（per-token min-keep=1 + min replicas 约束）
        assert abs(effective_dropped - rate) < 0.05, \
            f"rate={rate}: effective_dropped={effective_dropped:.3f}"
        print(f"OK rate={rate} effective={effective_dropped:.3f}")
```

- [ ] **Step 2.4** Main：

```python
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[skip] no CUDA"); sys.exit(0)
    test_drop_rate_zero_keeps_all()
    test_effective_drop_frac_matches()
    print("ALL OK")
```

- [ ] **Step 2.5** 跑：

```bash
cd /home/lzy/Artifact-Infer
/home/lzy/miniconda3/envs/vllm/bin/python -m eval.drop.tests.test_drop_invariants
```

Expected: `ALL OK`

- [ ] **Step 2.6** Commit

---

## Task 3 — `eval/drop/tier1_bench.py`

**Files**: Create `eval/drop/tier1_bench.py`（一个文件搞定）

**CLI**：

```
python -m eval.drop.tier1_bench \
    --output-dir eval_results/prefill_drop_l_sweep_tier1 \
    --t-local-values 1,8,64,512,2048 \
    --drop-rates 0.0,0.3 \
    --warmup-iters 10 \
    --iters 30 \
    --expert-overlap-path eval_results/tmp_phase3_overlap_test/moe_overlap_plan_smoke_plan_20260521_222623.json \
    --expert-overlap-strategy greedy_balance \
    --seed 42 \
    --wandb-project moe-drop-l-sweep
```

**职责**（线性结构，单文件 ~300 行）：

- [ ] **Step 3.1** 顶部 env setdefault + imports

```python
import os
os.environ.setdefault("MOE_DROP_IMPL", "gpu")
os.environ.setdefault("MOE_DROP_MIN_REPLICAS", "0")
os.environ.setdefault("MOE_DROP_GPU_STATS", "0")
os.environ.setdefault("MOE_PROFILE_OVERLAP_RUNTIME", "0")
os.environ.setdefault("MOE_RECORD_TIMING", "0")
```

- [ ] **Step 3.2** `parse_args()` —— argparse 8 个参数

- [ ] **Step 3.3** `setup_distributed()`：

```python
def setup_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    torch.set_default_device(f"cuda:{local_rank}")
    init_parallel_groups(tp_size=1, world_size=world, data_parallel_size=world, runtime_mode="owner_local_ep")
    return rank, local_rank, world
```

- [ ] **Step 3.4** `build_moe_block(args, world, max_T_local)`：构造单层 EP-HT MoE block + backend buffers + 权重初始化（normal_(std=0.02)）

- [ ] **Step 3.5** `make_router_logits(T_local, E_global, K, seed)` —— `torch.randn` fp32

- [ ] **Step 3.6** `run_one_cell(cell_id, T_local, drop_rate, modules, args)`：返回 dict 包含 per-repeat metric list

```python
def run_one_cell(cell_id, T_local, drop_rate, modules, hidden, logits, world, args):
    dispatch, experts, combine = modules
    # 重要：rate=0 时关 drop（重建 dispatch 太贵；改 attr 即可）
    dispatch.drop_rate = drop_rate
    dist.barrier()
    # warmup
    for _ in range(args.warmup_iters):
        tok_meta = dispatch(hidden, logits)
        eo = experts(tok_meta)
        _ = combine(eo, tok_meta)
    torch.cuda.synchronize()
    # measured
    rows = []
    for rep in range(args.iters):
        d_timer = CUDAEventTimer()
        e_timer = CUDAEventTimer()
        c_timer = CUDAEventTimer()
        t_timer = CUDAEventTimer()
        dist.barrier()
        with t_timer:
            with d_timer:
                tok_meta = dispatch(hidden, logits)
            with e_timer:
                eo = experts(tok_meta)
            with c_timer:
                out = combine(eo, tok_meta)
        torch.cuda.synchronize()
        d_us = d_timer.read(); e_us = e_timer.read(); c_us = c_timer.read(); t_us = t_timer.read()
        L_recv = int(tok_meta.recv_hidden.shape[0])
        L_send_kept = int(sum(tok_meta.send_counts))
        # all_gather rank-level metrics
        local = torch.tensor([d_us, e_us, c_us, t_us, float(L_recv), float(L_send_kept)],
                             dtype=torch.float64, device="cuda")
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        mat = torch.stack(gathered).cpu().numpy()  # [world, 6]
        rows.append({
            "cell_id": cell_id,
            "T_local": T_local,
            "drop_rate": drop_rate,
            "repeat_id": rep,
            "dispatch_us_rank_max": float(mat[:,0].max()),
            "experts_us_rank_max":  float(mat[:,1].max()),
            "combine_us_rank_max":  float(mat[:,2].max()),
            "total_us_rank_max":    float(mat[:,3].max()),
            "L_recv_mean": float(mat[:,4].mean()),
            "L_recv_max":  float(mat[:,4].max()),
            "L_recv_min":  float(mat[:,4].min()),
            "L_send_kept_mean": float(mat[:,5].mean()),
        })
    return rows
```

- [ ] **Step 3.7** Main：

```python
def main():
    args = parse_args()
    rank, local_rank, world = setup_distributed()
    seed_all(args.seed)

    T_locals = [int(x) for x in args.t_local_values.split(",")]
    rates = [float(x) for x in args.drop_rates.split(",")]
    cells = [(T, r) for T in T_locals for r in rates]
    # 随机化 cell 顺序（同 seed 保证跨 rank 一致）
    rng = random.Random(args.seed)
    rng.shuffle(cells)

    max_T = max(T_locals)
    modules = build_moe_block(args, world, max_T)

    jsonl_path = os.path.join(args.output_dir, "tier1_rows.jsonl")
    if rank == 0:
        # truncate
        os.makedirs(args.output_dir, exist_ok=True)
        open(jsonl_path, "w").close()
    dist.barrier()

    wandb_logger = WandbLogger(
        project=args.wandb_project, name=f"tier1_w{world}_seed{args.seed}",
        config=vars(args), enabled=args.wandb_project != "",
    )

    # 为每个 T_local 预生成 hidden + logits（同 seed 跨 cell）
    inputs_cache = {}
    for T in T_locals:
        gen = torch.Generator(device="cuda").manual_seed(args.seed * 100003 + T)
        h = torch.randn((T, args.hidden_size), dtype=torch.bfloat16, generator=gen, device="cuda")
        logits_gen = torch.Generator(device="cuda").manual_seed(args.seed * 7919 + T)
        l = torch.randn((T, args.num_experts), dtype=torch.float32, generator=logits_gen, device="cuda")
        inputs_cache[T] = (h, l)

    for ci, (T_local, drop_rate) in enumerate(cells):
        if rank == 0:
            print(f"[cell {ci+1}/{len(cells)}] T_local={T_local} drop_rate={drop_rate}")
        hidden, logits = inputs_cache[T_local]
        rows = run_one_cell(ci, T_local, drop_rate, modules, hidden, logits, world, args)
        if rank == 0:
            for r in rows:
                write_jsonl(jsonl_path, r)
                wandb_logger.log(r)
        # inter-cell cooldown
        torch.cuda.synchronize()
        dist.barrier()
        if ci != len(cells) - 1:
            time.sleep(3)

    dist.barrier()
    if rank == 0:
        write_summary(jsonl_path, args.output_dir)
    wandb_logger.finish()
    dist.destroy_process_group()
```

- [ ] **Step 3.8** `write_summary(jsonl_path, output_dir)`：读 JSONL → agg per cell → 算 `delta_pct`、`L*`、写 `tier1_summary.json`

```python
def write_summary(jsonl_path, output_dir):
    rows = [json.loads(l) for l in open(jsonl_path)]
    # group by (T_local, drop_rate) -> agg
    from collections import defaultdict
    cells = defaultdict(list)
    for r in rows:
        cells[(r["T_local"], r["drop_rate"])].append(r)
    cell_agg = {}
    for (T, rate), rs in cells.items():
        cell_agg[(T, rate)] = {
            "T_local": T, "drop_rate": rate,
            "total_us": agg_stats([r["total_us_rank_max"] for r in rs]),
            "dispatch_us": agg_stats([r["dispatch_us_rank_max"] for r in rs]),
            "experts_us": agg_stats([r["experts_us_rank_max"] for r in rs]),
            "combine_us": agg_stats([r["combine_us_rank_max"] for r in rs]),
            "L_recv_max": agg_stats([r["L_recv_max"] for r in rs]),
            "L_recv_mean": agg_stats([r["L_recv_mean"] for r in rs]),
            "L_send_kept_mean": agg_stats([r["L_send_kept_mean"] for r in rs]),
        }
    # 求 L* —— 用 total_us mean，alpha = -0.05
    T_locals = sorted({T for T, _ in cell_agg.keys()})
    delta_points = []
    for T in T_locals:
        base = cell_agg.get((T, 0.0))
        drop = cell_agg.get((T, 0.3))
        if base is None or drop is None: continue
        L = base["L_recv_max"]["p50"]
        delta = drop["total_us"]["mean"] - base["total_us"]["mean"]
        delta_pct = delta / base["total_us"]["mean"]
        ratio_experts = (
            (base["experts_us"]["mean"] - drop["experts_us"]["mean"])
            / max(base["total_us"]["mean"] - drop["total_us"]["mean"], 1e-9)
        )
        delta_points.append({
            "T_local": T, "L_recv_max_p50": L,
            "delta_pct": delta_pct, "delta_us": delta,
            "ratio_experts": ratio_experts,
            "baseline_total_us": base["total_us"]["mean"],
            "drop_total_us": drop["total_us"]["mean"],
        })
    L_star, status = compute_L_star(delta_points)
    h1_refuted = status != "no_crossing"
    h2_ratio = None
    if h1_refuted and delta_points:
        # 找最接近 L* 的点
        anchor = min(delta_points, key=lambda d: abs(d["L_recv_max_p50"] - (L_star or 0)))
        h2_ratio = anchor["ratio_experts"]

    # falsification floor: 最小 T_local 上 delta_pct > +2% ?
    small_dp = min(delta_points, key=lambda d: d["T_local"]) if delta_points else None
    floor_violated = small_dp is not None and small_dp["delta_pct"] > 0.02

    decision = (
        "STOP_INVESTIGATE_DROP_IMPL" if floor_violated
        else "RUN_TIER2" if h1_refuted
        else "STOP_CLOSED_NEGATIVE"
    )
    summary = {
        "L_star": L_star,
        "L_star_status": status,
        "h1_status": "refuted" if h1_refuted else "not_refuted",
        "h2_ratio_experts": h2_ratio,
        "decision": decision,
        "pre_reg_check": {
            "small_L_drop_overhead_pct": small_dp["delta_pct"] if small_dp else None,
            "falsification_floor_violated": floor_violated,
        },
        "delta_points": delta_points,
    }
    with open(os.path.join(output_dir, "tier1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] decision={decision} L_star={L_star} h1={summary['h1_status']}")
```

- [ ] **Step 3.9** Commit

---

## Task 4 — 1 卡 smoke

跑：

```bash
cd /home/lzy/Artifact-Infer
torchrun --nproc_per_node=1 --master_port=29500 \
    -m eval.drop.tier1_bench \
    --output-dir /tmp/tier1_smoke_1rank \
    --t-local-values 8,64 \
    --drop-rates 0.0,0.3 \
    --warmup-iters 2 \
    --iters 3 \
    --expert-overlap-path "" \
    --wandb-project ""
```

Pass：
- 没有 traceback
- `/tmp/tier1_smoke_1rank/tier1_rows.jsonl` 行数 = 2 (T) × 2 (rate) × 3 (iter) = 12
- `tier1_summary.json` 存在，`decision ∈ {STOP_*, RUN_TIER2}`

注：1 卡 EP 是退化 case，可能 routing 全 local；这一步只验证脚本 pipeline。

---

## Task 5 — 8 卡 smoke + 主 sweep

**Smoke**（5 分钟）：

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
    -m eval.drop.tier1_bench \
    --output-dir /tmp/tier1_smoke_8rank \
    --t-local-values 8,64 \
    --drop-rates 0.0,0.3 \
    --warmup-iters 2 \
    --iters 3 \
    --expert-overlap-path eval_results/tmp_phase3_overlap_test/moe_overlap_plan_smoke_plan_20260521_222623.json
```

Pass：`rows.jsonl` 12 行，`tier1_summary.json` decision 合理。

**主 sweep**（~30-60 min）：

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
    -m eval.drop.tier1_bench \
    --output-dir eval_results/prefill_drop_l_sweep_tier1 \
    --t-local-values 1,8,64,512,2048 \
    --drop-rates 0.0,0.3 \
    --warmup-iters 10 \
    --iters 30 \
    --expert-overlap-path eval_results/tmp_phase3_overlap_test/moe_overlap_plan_smoke_plan_20260521_222623.json \
    --wandb-project moe-drop-l-sweep
```

Pass：
- JSONL 5 × 2 × 30 = 300 行
- `tier1_summary.json.decision` 输出 closed-negative / run_tier2 / stop_investigate

---

## Task 6 — `eval/drop/tier2_bench.py`（如 Tier 1 决定 RUN_TIER2 才写）

**触发条件**：`tier1_summary.json["decision"] == "RUN_TIER2"`

**CLI**：

```bash
torchrun --nproc_per_node=8 -m eval.drop.tier2_bench \
    --tier1-summary eval_results/prefill_drop_l_sweep_tier1/tier1_summary.json \
    --model ~/models/Qwen3-30B-A3B \
    --output-dir eval_results/prefill_drop_l_sweep_tier2 \
    --moe-impl ep_ht --runtime-mode owner_local_ep \
    --expert-overlap-path <same> --expert-overlap-strategy greedy_balance \
    --repeats 3 --warmup 1
```

**职责**：
1. 读 tier1_summary.json 提取 L_star，算 target prompt_tokens
2. 构造 (batch=4, prompt_len=L_star/4) synthetic prompts (`token_id = " the"` ×N)
3. 跑 baseline 和 drop 各 3 reps + 1 warmup
4. 从 `engine.last_generation_metrics` 读 `prefill_time_s`, `prefill_throughput_tok_s`, `e2e_total_time_s`
5. 写 `tier2_rows.jsonl` + `tier2_summary.json`

详细实现等 Tier 1 跑完看结果再写——如果 Tier 1 是 closed-negative，跳过 Task 6。

---

## Task 7 — 写最终结论

**路径 B（closed-negative）**：写 `docs/claude-moe/drop/0526-plan/0526-final_report.md` 包含：
- Tier 1 主图（delta_pct vs L_recv_max）
- H1/H2 verdict
- 与 v1/v2/K_eff 三个负实验并排表
- 用户两个问题的直接回答

**路径 A**：加 Tier 2 的 e2e 验证 + Phase 5 ablation 草稿。

---

## 自检

- ✅ TDD 顺序：tests → bench → smoke → 主 sweep
- ✅ 所有阈值（warmup ≥ 10, iters ≥ 30, mean/std/p99）满足
- ✅ CUDA event only，无 time.time()
- ✅ JSONL 每行 (L, drop_rate, repeat_id, ...)
- ✅ W&B optional，无配置时静默 disable
- ✅ standalone python -m runnable
- ✅ Unit test 直接打到 `apply_drop_gpu_simple`，单卡可跑
- ⚠️ Tier 2 详细 task 等 Tier 1 结果再展开

---

## Hard time-box

Day 1 内完成 Task 1-5（含主 sweep），Day 2 视结果做 Task 6 或 Task 7。
