import os
from nanovllm import LLM
from nanovllm.config import Config
# 假设你的 SamplingParams 在这，如果不对请替换为你的实际导入路径
from nanovllm import SamplingParams 
from transformers import AutoTokenizer

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    # 构造 3 个长度差异巨大的 Prompt
    prompts = [
        # Prompt 1: 极短文本 (快速进入 Decode，测试短序列的生成与尽早结束)
        "What is the capital of Japan? Please answer in exactly one word.",
        
        # Prompt 2: 中等文本 (代码生成，测试逻辑连贯性和中等长度的 Chunk 处理)
        "Write a concise Python function to compute the Fibonacci sequence up to n using dynamic programming. Include comments.",
        
        # Prompt 3: 较长文本 (硬核系统知识，不仅能测出多个 Chunk 的切分，还非常契合你现在的开发方向！)
        "Explain the concept of Paged Attention in Large Language Models. How does it manage KV cache memory, and why is it better than traditional continuous memory allocation for LLM inference? Please provide a detailed, step-by-step technical breakdown."
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    sampling_params = SamplingParams(
        temperature=0.6, 
        max_tokens=64 # 设为 20，足够我们观察 Decode 行为，又不会跑太久
    )

    # 初始化 LLM，关键在于压低 token budget，强行逼迫 Scheduler 进行切块
    # 这里的参数根据你的 Config 实际传参方式进行调整
    llm = LLM(
        path, 
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_batched_tokens=32, # 【核心参数】：将每轮调度的 Budget 压低到 32
        max_num_seqs=4,            # 允许同时处理 4 个请求
        enforce_eager=True         # 调试阶段依然保持 Eager 模式，方便看探针
    )

    print("\n==================================================")
    print("开始多请求混合调度生成测试...")
    print("==================================================\n")

    outputs = llm.generate(prompts, sampling_params)

    print("\n==================================================")
    print("最终生成结果:")
    for i, output in enumerate(outputs):
        # 假设 output 的结构包含 text 属性，请根据你的实际结构调整
        print(f"\n[Prompt {i+1}]: {prompts[i][:30]}...")
        print(f"[Output {i+1}]: {output.text if hasattr(output, 'text') else output}")
    print("==================================================\n")

if __name__ == "__main__":
    main()