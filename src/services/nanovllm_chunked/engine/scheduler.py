from collections import deque

from ..config import Config
from .sequence import Sequence, SequenceStatus
from .block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> list[tuple[Sequence, int]]:
        token_budget = self.max_num_batched_tokens
        scheduled_seqs: list[tuple[Sequence, int]] = []

        # 优先调度 RUNNING 队列
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            seq = self.running[req_index]
            
            num_new_tokens = min(seq.num_tokens - seq.num_computed_tokens, token_budget)
            print(f"num_new_tokens: {num_new_tokens}")
            allocated = False
            
            while True:
                if self.block_manager.allocate_slots(seq, num_new_tokens):
                    allocated = True
                    break
                
                # 显存不足！直接从 self.running 队端(尾部)弹出优先级最低的请求
                preempted_seq = self.running.pop()
                self.preempt(preempted_seq)
                
                # 如果弹出的刚好是自己，说明所有比自己优先级低的都被踢光了，
                # 显存还是不够。那么自己也被踢出了 running 队列，无法调度。
                if preempted_seq == seq:
                    break
            
            if allocated:
                # 申请成功，记录元数据，扣减 budget
                scheduled_seqs.append((seq, num_new_tokens))
                token_budget -= num_new_tokens
                # 指针移动到下一个请求
                req_index += 1
            
            # 如果 allocated 为 False，说明 seq 已经被 pop 掉了。
            # 此时 self.running 长度减 1，当前位置上的元素变成了原本的下一个元素。
            # 所以 req_index 不需要 +1，下一轮循环会自动处理接替上来的新请求
    
        # 如果还有剩余额度，调度 WAITING 队列
        while self.waiting and token_budget > 0 and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            num_new_tokens = min(seq.num_tokens - seq.num_computed_tokens, token_budget)
            
            if self.block_manager.allocate_slots(seq, num_new_tokens):
                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_seqs.append((seq, num_new_tokens))
                token_budget -= num_new_tokens
            else:
                break 

        # ----------------------------------------------------
        # 3. 推进进度与判定
        # ----------------------------------------------------
        for seq, num_new_tokens in scheduled_seqs:
            seq.num_computed_tokens += num_new_tokens
            seq.is_prefill_chunk = seq.num_computed_tokens < seq.num_tokens

            # 【确认加上这行打印！】
            print(f"[Scheduler] Seq {seq.seq_id} | "
                  f"Allocated chunk: {num_new_tokens} | "
                  f"Progress: {seq.num_computed_tokens}/{seq.num_tokens} | "
                  f"is_prefill_chunk: {seq.is_prefill_chunk}")

        return scheduled_seqs

    def preempt(self, seq: Sequence):
        # seq.status = SequenceStatus.WAITING
        seq.reset_for_preemption()
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            # 识别被拦截的中间块，跳过追加
            if token_id is None or token_id == -1:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
