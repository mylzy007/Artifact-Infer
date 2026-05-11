"""Smoke test the LLMEngine wiring on a small dense model (Qwen3-4B).

Validates Scheduler <-> BlockManager <-> ModelRunner method propagation
through the orchestrator. No MoE involved here — that's _test_engine_moe.py.
"""
import os
import sys
import time

import torch

CKPT = "/home/yyx/models/Qwen3-4B"


def main():
    from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine
    from workshop.nanovllm_moe.services.sampling_params import SamplingParams

    print(f"Constructing LLMEngine on {CKPT} (enforce_eager=True)...", flush=True)
    t0 = time.time()
    engine = LLMEngine(
        model=CKPT,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        max_model_len=512,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
    )
    print(f"  built in {time.time() - t0:.1f}s", flush=True)

    prompts = [
        "The capital of France is",
        "Once upon a time, there was",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)

    print("Generating...", flush=True)
    out = engine.generate(prompts, sp, use_tqdm=False)
    for i, item in enumerate(out):
        print(f"--- prompt {i} ---")
        print(f"prompt: {prompts[i]!r}")
        print(f"reply : {item['text']!r}")
        print(f"toks  : {item['token_ids']}")

    assert len(out) == len(prompts)
    for i, item in enumerate(out):
        assert len(item["token_ids"]) == 16, (
            f"prompt {i}: expected 16 generated tokens, got {len(item['token_ids'])}"
        )
    print("OK: LLMEngine end-to-end works on dense Qwen3-4B")


if __name__ == "__main__":
    main()
