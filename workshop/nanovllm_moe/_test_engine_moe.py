"""End-to-end MoE engine test on Qwen3-30B-A3B.

The full Qwen3-30B-A3B has 48 layers (~60 GB bf16) which doesn't fit on one RTX 4090.
We trim to NUM_LAYERS layers via Config.num_hidden_layers_override — this loses the
"text quality" property of the test (the truncated model is no longer a real LM) but
preserves the property we *do* care about here: that the LLMEngine end-to-end pipeline
(Scheduler -> BlockManager -> ModelRunner -> Qwen3MoeForCausalLM with FusedMoE backed by
MoeBackend, including dispatch/experts/combine kernels) runs to completion without errors
and produces well-formed token streams.

For "real text quality" validation we already have:
  - _test_parity_hf.py / _test_parity_hf_triton.py — block-level numerical match vs HF
  - _test_cudagraph.py — bit-exact match between eager and CUDA-graph replay

This test fills the gap between those: the *engine* layer (scheduler + block manager +
prefill + decode + sampling).

Run with:
  MOE_IMPL=torch python -m workshop.nanovllm_moe._test_engine_moe
  MOE_IMPL=triton python -m workshop.nanovllm_moe._test_engine_moe
  ENFORCE_EAGER=0 MOE_IMPL=triton python -m workshop.nanovllm_moe._test_engine_moe
"""
import os
import time

import torch

CKPT = "/home/yyx/models/Qwen3-30B-A3B"
MOE_IMPL = os.environ.get("MOE_IMPL", "triton")
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") == "1"
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "8"))


def main():
    from workshop.nanovllm_moe.services.engine.llm_engine import LLMEngine
    from workshop.nanovllm_moe.services.sampling_params import SamplingParams

    print(f"Constructing LLMEngine on {CKPT}", flush=True)
    print(f"  moe_impl={MOE_IMPL!r}, enforce_eager={ENFORCE_EAGER}, "
          f"num_layers={NUM_LAYERS} (trimmed from 48)", flush=True)
    t0 = time.time()
    engine = LLMEngine(
        model=CKPT,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        max_model_len=512,
        enforce_eager=ENFORCE_EAGER,
        gpu_memory_utilization=0.85,
        moe_impl=MOE_IMPL,
        num_hidden_layers_override=NUM_LAYERS,
    )
    print(f"  built in {time.time() - t0:.1f}s", flush=True)
    print(f"  free GPU mem: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB", flush=True)

    prompts = [
        "The capital of France is",
        "Once upon a time, there was",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)

    print("Generating...", flush=True)
    t0 = time.time()
    out = engine.generate(prompts, sp, use_tqdm=False)
    elapsed = time.time() - t0
    n_gen = sum(len(o["token_ids"]) for o in out)
    print(f"  done in {elapsed:.2f}s ({n_gen / elapsed:.1f} tok/s aggregate)", flush=True)

    for i, item in enumerate(out):
        print(f"--- prompt {i} ---")
        print(f"prompt: {prompts[i]!r}")
        print(f"reply (probably gibberish at {NUM_LAYERS}/48 layers): {item['text']!r}")
        print(f"first 8 ids: {item['token_ids'][:8]}")

    assert len(out) == len(prompts)
    for i, item in enumerate(out):
        toks = item["token_ids"]
        assert len(toks) == 32, (
            f"prompt {i}: expected 32 generated tokens, got {len(toks)}"
        )
        # Token IDs must be in the vocab. EOS-and-below is allowed if ignore_eos was respected.
        assert all(0 <= t < engine.tokenizer.vocab_size for t in toks), (
            f"prompt {i}: out-of-range token id: min={min(toks)}, max={max(toks)}, "
            f"vocab={engine.tokenizer.vocab_size}"
        )
        # A trimmed-but-working model often outputs a constant or near-constant id;
        # we only check that we don't produce all the same id (which would suggest
        # the model collapsed to a single logit).
        assert len(set(toks)) > 1, (
            f"prompt {i}: all generated tokens are the same id {toks[0]}, suggests broken pipeline"
        )

    print(f"OK: LLMEngine end-to-end works on Qwen3-30B-A3B "
          f"(moe_impl={MOE_IMPL}, layers={NUM_LAYERS}, enforce_eager={ENFORCE_EAGER})")


if __name__ == "__main__":
    main()
