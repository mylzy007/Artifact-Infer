#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Capture per-layer MoE routing decisions from a Qwen3 / Qwen3.5 checkpoint into
a torch trace file consumable by
``tests/kernels/moe/modular_kernel_tools/benchmark_eplb_multigpu.py``
via its ``--routing-trace`` flag.

Two capture modes are provided:

1. ``hf`` (HuggingFace full-model hook)
   - Loads the model with HuggingFace ``AutoModelForCausalLM`` (with
     ``device_map="auto"``), patches every MoE block's forward to record
     ``(selected_experts, routing_weights)``, and runs a single prefill on
     a sample prompt.
   - This gives "real" routing decisions across all layers because each
     layer sees the actual hidden_states that would flow through during
     inference.
   - Requires the model architecture to be supported by the installed
     transformers version. Tested on ``Qwen3-30B-A3B`` (qwen3_moe) with
     transformers 4.57+.

2. ``router-only``
   - Reads gate weights directly from the safetensors index, then runs
     just the gate -> softmax -> topk pipeline on a synthetic hidden
     stream (Gaussian after layer-norm scaling).
   - Useful when transformers does not yet recognise the architecture
     (e.g. Qwen3.5 with transformers 4.57.x). The trained expert
     preferences are real; only the per-layer hidden_states are stylised.

The output file is a torch ``.pt`` blob with the schema documented in
``benchmark_eplb_multigpu._load_routing_trace``.

Example:

    # Full HF capture (Qwen3-30B-A3B, ~60 GB params spread across 8x4090):
    python capture_routing_trace.py \\
        --model /home/yyx/models/Qwen3-30B-A3B \\
        --mode hf \\
        --prompt "Write a short essay on coffee." \\
        --max-tokens 1024 \\
        --output /home/yyx/personal/inference/vllm-bench/traces/qwen3.pt

    # Router-only capture (Qwen3.5-35B-A3B):
    python capture_routing_trace.py \\
        --model /home/yyx/models/Qwen3.5-35B-A3B \\
        --mode router-only \\
        --num-tokens 8192 \\
        --output /home/yyx/personal/inference/vllm-bench/traces/qwen3p5.pt
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class MoEDims:
    model_name: str
    num_layers: int
    hidden_size: int
    num_experts: int
    topk: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    rms_norm_eps: float


def _read_config(model_dir: Path) -> MoEDims:
    cfg_path = model_dir / "config.json"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    text = cfg.get("text_config", cfg)

    arches = cfg.get("architectures") or []
    model_name = arches[0] if arches else cfg.get("model_type", "custom")

    required = (
        "hidden_size",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "num_hidden_layers",
    )
    missing = [k for k in required if k not in text]
    if missing:
        raise ValueError(
            f"{cfg_path} is missing MoE fields: {missing}. "
            "Got keys: " + ", ".join(sorted(text.keys()))
        )

    num_layers = int(text["num_hidden_layers"])
    num_layers += int(text.get("mtp_num_hidden_layers", 0))

    return MoEDims(
        model_name=str(model_name),
        num_layers=num_layers,
        hidden_size=int(text["hidden_size"]),
        num_experts=int(text["num_experts"]),
        topk=int(text["num_experts_per_tok"]),
        moe_intermediate_size=int(text["moe_intermediate_size"]),
        norm_topk_prob=bool(text.get("norm_topk_prob", True)),
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
    )


# ---------------------------------------------------------------------------
# Mode 1: HuggingFace full-model hook
# ---------------------------------------------------------------------------


def capture_hf(
    model_dir: Path,
    dims: MoEDims,
    prompt: str,
    max_tokens: int,
    dtype: torch.dtype,
    device_map: str,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return (per_layer_topk_ids, per_layer_topk_weights), each [N, topk]
    on CPU, captured during a single prefill of ``prompt`` truncated to
    ``max_tokens`` tokens."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[hf] loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    print(f"[hf] loading model from {model_dir} (dtype={dtype}, device_map={device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    captured_ids: dict[int, torch.Tensor] = {}
    captured_wts: dict[int, torch.Tensor] = {}

    moe_blocks = []
    decoder_layers = None
    for module in model.modules():
        cls_name = module.__class__.__name__
        if "SparseMoeBlock" in cls_name or cls_name.endswith("MoeBlock"):
            moe_blocks.append(module)
        if cls_name.endswith("DecoderLayer") and decoder_layers is None:
            pass

    if not moe_blocks:
        # Fallback: walk model.model.layers and pick out blocks named "mlp"
        # whose class contains "MoE" or "Sparse".
        try:
            layers = model.model.layers  # type: ignore[attr-defined]
        except AttributeError:
            try:
                layers = model.model.model.layers  # type: ignore[attr-defined]
            except AttributeError as e:
                raise RuntimeError(
                    "Could not locate MoE blocks in the model. Tried "
                    "model.modules() match and model.model.layers walk."
                ) from e
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            cls_name = mlp.__class__.__name__
            if "MoE" in cls_name or "Moe" in cls_name or "Sparse" in cls_name:
                moe_blocks.append(mlp)

    print(f"[hf] found {len(moe_blocks)} MoE blocks (expected ~{dims.num_layers})")
    if len(moe_blocks) == 0:
        raise RuntimeError("Found 0 MoE blocks in the model")

    def _make_hook(layer_idx: int):
        def _hook(module, args, kwargs, output):
            # Many SparseMoE blocks return (hidden, router_logits). We need
            # the topk decisions, which we recompute from router_logits to
            # avoid relying on internal attributes.
            if isinstance(output, tuple) and len(output) >= 2:
                router_logits = output[1]
            else:
                # Some blocks expose .last_router_logits or similar.
                router_logits = getattr(module, "_last_router_logits", None)
                if router_logits is None:
                    raise RuntimeError(
                        f"Layer {layer_idx} ({module.__class__.__name__}) "
                        "did not return router_logits in its forward output. "
                        "This capture mode only supports MoE blocks whose "
                        "forward returns (hidden, router_logits)."
                    )

            # router_logits: [num_tokens, num_experts]
            # Recreate the same topk routing the model uses.
            with torch.no_grad():
                rl = router_logits.to(torch.float32)
                if rl.dim() == 3:
                    rl = rl.reshape(-1, rl.shape[-1])
                routing_probs = torch.softmax(rl, dim=-1)
                topk = min(dims.topk, routing_probs.shape[-1])
                weights, ids = torch.topk(routing_probs, k=topk, dim=-1)
                if dims.norm_topk_prob:
                    weights = weights / weights.sum(dim=-1, keepdim=True)
            captured_ids[layer_idx] = ids.detach().to("cpu", dtype=torch.long).contiguous()
            captured_wts[layer_idx] = (
                weights.detach().to("cpu", dtype=torch.float32).contiguous()
            )

        return _hook

    handles = []
    for idx, blk in enumerate(moe_blocks):
        handles.append(blk.register_forward_hook(_make_hook(idx), with_kwargs=True))

    print(f"[hf] tokenising prompt and truncating to {max_tokens} tokens")
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_tokens
    )
    input_ids = enc["input_ids"]
    if input_ids.shape[-1] < max_tokens:
        # repeat to pad up to max_tokens
        reps = (max_tokens + input_ids.shape[-1] - 1) // input_ids.shape[-1]
        input_ids = input_ids.repeat(1, reps)[:, :max_tokens]
    input_ids = input_ids.to(model.device if hasattr(model, "device") else "cuda:0")
    attention_mask = torch.ones_like(input_ids)

    print(f"[hf] running prefill on {input_ids.shape[-1]} tokens")
    torch.manual_seed(seed)
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    for h in handles:
        h.remove()

    if not captured_ids:
        raise RuntimeError("Captured 0 layers; hooks were not triggered.")

    layer_indices = sorted(captured_ids.keys())
    print(f"[hf] captured {len(layer_indices)} layers from {layer_indices[0]} .. {layer_indices[-1]}")

    ids_list = [captured_ids[i] for i in layer_indices]
    wts_list = [captured_wts[i] for i in layer_indices]
    return ids_list, wts_list


# ---------------------------------------------------------------------------
# Mode 2: router-only (for archs not yet in transformers, e.g. Qwen3.5)
# ---------------------------------------------------------------------------


def _layer_index_from_key(key: str) -> tuple[int, int] | None:
    """Return ``(group, layer_within_group)`` for a key like ``foo.layers.<i>.bar``.

    The ``group`` lets us distinguish two layer-0 gates that live in
    different sub-modules (e.g. main backbone ``layers.0`` vs. MTP head
    ``mtp.layers.0``) so we don't accidentally overwrite one with the other
    when collecting gates.
    """
    parts = key.split(".")
    for j, p in enumerate(parts):
        if p == "layers" and j + 1 < len(parts):
            try:
                idx = int(parts[j + 1])
            except ValueError:
                return None
            # Use the prefix path before ".layers." as the group identifier.
            prefix = ".".join(parts[:j])
            # Heuristic ranking: main backbone first (no special tag),
            # then "mtp", then anything else; alphabetical for stability.
            if "mtp" in prefix.lower():
                group = 1
            elif "vision" in prefix.lower() or "visual" in prefix.lower():
                group = 9
            else:
                group = 0
            return (group, idx)
    return None


def _gate_weight_keys(weight_map: dict[str, str]) -> list[tuple[int, str, str]]:
    """Return [(layer_idx, weight_key, shard_filename), ...] sorted globally.

    Layer indices are renumbered 0..N-1 across all groups (e.g. main + MTP).
    """
    raw: list[tuple[tuple[int, int], str, str]] = []
    for key, shard in weight_map.items():
        # qwen3_moe / qwen3_5_moe both name the router as ``...mlp.gate.weight``
        # (no bias). Be liberal: accept "gate.weight" inside an mlp block.
        if not key.endswith("gate.weight"):
            continue
        if "shared_expert_gate" in key:
            continue  # those are 1-channel, not the routing gate
        if "experts." in key:
            # "experts.{i}.gate_proj.weight" and similar — not router gates
            continue
        if "vision" in key.lower() or "visual" in key.lower():
            continue
        if ".mlp." not in key:
            continue
        sort_key = _layer_index_from_key(key)
        if sort_key is None:
            continue
        raw.append((sort_key, key, shard))
    raw.sort(key=lambda t: t[0])
    return [(i, key, shard) for i, (_, key, shard) in enumerate(raw)]


def capture_router_only(
    model_dir: Path,
    dims: MoEDims,
    num_tokens: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"safetensors index not found at {index_path}")
    with index_path.open(encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    if not weight_map:
        raise RuntimeError(f"No weight_map in {index_path}")

    gate_keys = _gate_weight_keys(weight_map)
    if not gate_keys:
        raise RuntimeError(
            "Could not find any '...mlp.gate.weight' in the safetensors "
            "index. Inspect a few keys: "
            + ", ".join(list(weight_map.keys())[:8])
        )
    print(
        f"[router-only] found {len(gate_keys)} candidate gate weights "
        f"(expected up to {dims.num_layers}). First three: "
        + ", ".join(k for _, k, _ in gate_keys[:3])
    )

    # Group by shard so we open each file once.
    by_shard: dict[str, list[tuple[int, str]]] = {}
    for layer_idx, key, shard in gate_keys:
        by_shard.setdefault(shard, []).append((layer_idx, key))

    gates: dict[int, torch.Tensor] = {}
    for shard, items in by_shard.items():
        shard_path = model_dir / shard
        if not shard_path.exists():
            # accept the alt naming "model.safetensors-NN-of-MM.safetensors"
            alt = list(model_dir.glob(shard + "*"))
            if alt:
                shard_path = alt[0]
        if not shard_path.exists():
            raise FileNotFoundError(f"safetensors shard missing: {shard_path}")
        with safe_open(str(shard_path), framework="pt", device="cpu") as st:
            for layer_idx, key in items:
                w = st.get_tensor(key).to(dtype)
                if w.dim() != 2:
                    raise ValueError(
                        f"Expected 2D gate weight at {key}, got shape {tuple(w.shape)}"
                    )
                if w.shape[0] != dims.num_experts or w.shape[1] != dims.hidden_size:
                    raise ValueError(
                        f"Gate {key} has shape {tuple(w.shape)} but config "
                        f"expects ({dims.num_experts}, {dims.hidden_size})"
                    )
                gates[layer_idx] = w

    layer_indices = sorted(gates.keys())
    if len(layer_indices) < dims.num_layers:
        print(
            f"[router-only] WARNING: only found gates for {len(layer_indices)}/"
            f"{dims.num_layers} layers (this is fine if some layers are dense)"
        )
    print(f"[router-only] using {len(layer_indices)} layers' gates")

    # Build a stylised hidden-state stream that loosely mimics post layer-norm
    # bf16 activations: zero-mean, unit-variance Gaussian per token. We
    # propagate the same hidden_states through each layer's gate (i.e. we
    # ignore the cross-layer hidden_state evolution). The gate softmax
    # therefore reflects which experts the trained router prefers for
    # generic activations, not per-layer-specific ones. This is a
    # deliberate trade-off so we can capture without loading the full
    # transformer body.
    torch.manual_seed(seed)
    hidden = torch.randn(
        (num_tokens, dims.hidden_size), device=device, dtype=torch.float32
    )

    ids_list: list[torch.Tensor] = []
    wts_list: list[torch.Tensor] = []
    for layer_idx in layer_indices:
        gate = gates[layer_idx].to(device=device, dtype=torch.float32)
        # router_logits = hidden @ gate.T
        logits = hidden @ gate.t()
        probs = torch.softmax(logits, dim=-1)
        weights, ids = torch.topk(probs, k=dims.topk, dim=-1)
        if dims.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        ids_list.append(ids.detach().to("cpu", dtype=torch.long).contiguous())
        wts_list.append(
            weights.detach().to("cpu", dtype=torch.float32).contiguous()
        )

    return ids_list, wts_list


# ---------------------------------------------------------------------------
# Wrapper / IO
# ---------------------------------------------------------------------------


def _save_trace(
    out_path: Path,
    dims: MoEDims,
    ids_list: list[torch.Tensor],
    wts_list: list[torch.Tensor],
    extra: dict | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not ids_list:
        raise RuntimeError("Refusing to save empty trace")
    global_n = ids_list[0].shape[0]

    layers_blob = []
    for li, (ids, wts) in enumerate(zip(ids_list, wts_list)):
        if ids.shape[0] != global_n or wts.shape[0] != global_n:
            raise RuntimeError(
                f"Layer {li}: ids {tuple(ids.shape)} and wts {tuple(wts.shape)} "
                f"have inconsistent token counts (expected {global_n})"
            )
        layers_blob.append({"layer_idx": li, "topk_ids": ids, "topk_weights": wts})

    metadata = {
        "model_name": dims.model_name,
        "num_layers": len(layers_blob),
        "num_experts": dims.num_experts,
        "topk": dims.topk,
        "hidden_size": dims.hidden_size,
        "moe_intermediate_size": dims.moe_intermediate_size,
        "global_num_tokens": global_n,
    }
    if extra:
        metadata.update(extra)

    torch.save({"metadata": metadata, "layers": layers_blob}, out_path)
    print(f"[save] wrote {out_path} ({len(layers_blob)} layers, {global_n} tokens each)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="HF model dir")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hf", "router-only"],
        required=True,
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a long story about a robot that learns to paint.",
        help="(hf mode) prompt to tokenise; padded/truncated to --max-tokens",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help=(
            "(hf mode) read the prompt text from this file instead of "
            "--prompt. Useful for long, multi-line workloads such as the "
            "concatenated AIME prefill produced by "
            "generate_aime_responses.py."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="(hf mode) target prefill token count",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=8192,
        help="(router-only mode) synthetic token count",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="(hf mode) accelerate device_map; 'auto' spreads across visible GPUs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="(router-only mode) device for the small router compute",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    dims = _read_config(model_dir)
    print(
        f"[config] {dims.model_name}: layers={dims.num_layers}, "
        f"hidden={dims.hidden_size}, experts={dims.num_experts}, "
        f"topk={dims.topk}, moe_int={dims.moe_intermediate_size}"
    )

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]

    if args.mode == "hf":
        if args.prompt_file:
            prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
            prompt_provenance = f"file:{args.prompt_file}"
            print(
                f"[hf] using prompt from {args.prompt_file} "
                f"({len(prompt_text)} chars)"
            )
        else:
            prompt_text = args.prompt
            prompt_provenance = "cli:--prompt"
        ids_list, wts_list = capture_hf(
            model_dir=model_dir,
            dims=dims,
            prompt=prompt_text,
            max_tokens=args.max_tokens,
            dtype=dtype,
            device_map=args.device_map,
            seed=args.seed,
        )
        _save_trace(
            Path(args.output),
            dims,
            ids_list,
            wts_list,
            extra={
                "capture_mode": "hf",
                "prompt_tokens": args.max_tokens,
                "prompt_provenance": prompt_provenance,
                "prompt_excerpt": prompt_text[:200],
            },
        )
    else:
        ids_list, wts_list = capture_router_only(
            model_dir=model_dir,
            dims=dims,
            num_tokens=args.num_tokens,
            dtype=dtype,
            seed=args.seed,
            device=args.device,
        )
        _save_trace(
            Path(args.output),
            dims,
            ids_list,
            wts_list,
            extra={
                "capture_mode": "router-only",
                "synthetic_tokens": args.num_tokens,
            },
        )


if __name__ == "__main__":
    main()
