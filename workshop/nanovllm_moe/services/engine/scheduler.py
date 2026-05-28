from collections import deque

from ..config import Config
from .sequence import Sequence, SequenceStatus
from src.core.service import BaseService


class Scheduler(BaseService):
    def __init__(self, config: Config):
        super().__init__()
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        prefill_slots = max(0, self.max_num_seqs - len(self.running))
        while self.waiting and num_seqs < prefill_slots:
            seq = self.waiting[0]
            if len(seq) > self.max_model_len:
                raise RuntimeError(
                    f"prompt length {len(seq)} exceeds max_model_len={self.max_model_len}"
                )
            if len(seq) > self.max_num_batched_tokens:
                raise RuntimeError(
                    f"prompt length {len(seq)} exceeds max_num_batched_tokens="
                    f"{self.max_num_batched_tokens}"
                )
            uncached_tokens = len(seq) - seq.num_cached_tokens
            if num_batched_tokens + uncached_tokens > self.max_num_batched_tokens:
                break
            if not self.can_allocate(seq):
                break
            num_seqs += 1
            self.allocate(seq)
            num_batched_tokens += uncached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            if len(seq) >= self.max_model_len:
                seq.status = SequenceStatus.FINISHED
                self.deallocate(seq)
                continue
            while not self.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            raise RuntimeError(
                "scheduler could not schedule any sequence; check KV cache capacity "
                f"(running={len(self.running)}, waiting={len(self.waiting)}, "
                f"max_num_seqs={self.max_num_seqs})"
            )
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            if len(seq) >= self.max_model_len:
                seq.status = SequenceStatus.FINISHED
                self.deallocate(seq)
                self.running.remove(seq)
                continue
            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
                or len(seq) >= self.max_model_len
            ):
                seq.status = SequenceStatus.FINISHED
                self.deallocate(seq)
                self.running.remove(seq)
