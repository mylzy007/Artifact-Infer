from collections import deque
import xxhash
import numpy as np

from .sequence import Sequence

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 流式分配：取代 can_allocate, allocate, can_append, may_append
    def allocate_slots(self, seq: Sequence, num_new_tokens: int) -> bool:
        """
        根据本次调度的 token 数量，按需流式分配物理块。
        如果物理块不足，直接返回 False 触发抢占。
        """

        target_tokens = seq.num_computed_tokens + num_new_tokens
        target_blocks = (target_tokens + self.block_size - 1) // self.block_size
        
        # 当前已经持有了多少个物理块
        current_blocks = len(seq.block_table)
        blocks_needed = target_blocks - current_blocks
        
        # 如果空闲块不够，直接拒绝分配 (Scheduler 会去抢占别人)
        if blocks_needed > len(self.free_block_ids):
            return False

        if not hasattr(seq, '_last_hash'):
            seq._last_hash = -1

        # 2. 检查之前未满的最后一个块，现在是否已经被填满了？
        if current_blocks > 0:
            last_block_idx = current_blocks - 1
            last_block_id = seq.block_table[last_block_idx]
            last_block = self.blocks[last_block_id]
            token_ids = seq.block(last_block_idx)
            
            # 如果刚刚填满，且还没有算过哈希，把它加入 Prefix Cache
            if len(token_ids) == self.block_size and last_block.hash == -1:
                prefix_hash = -1
                if last_block_idx > 0:
                    prev_block_id = seq.block_table[last_block_idx - 1]
                    prefix_hash = self.blocks[prev_block_id].hash
                    
                h = self.compute_hash(token_ids, prefix_hash)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
                seq._last_hash = h

        # 3. 为本次新增的块进行分配 (支持前缀缓存命中)
        for i in range(current_blocks, target_blocks):
            token_ids = seq.block(i)
            # 只有完整的块才有资格计算 hash 并参与 Prefix Cache
            is_full_block = len(token_ids) == self.block_size
            
            if is_full_block:
                h = self.compute_hash(token_ids, seq._last_hash)
                block_id = self.hash_to_block_id.get(h, -1)
                
                # 命中前缀缓存！
                if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                    if block_id in self.used_block_ids:
                        self.blocks[block_id].ref_count += 1
                    else:
                        # 块在 free 队列中，但数据还健在，重新激活它
                        self._allocate_block(block_id)
                        self.blocks[block_id].update(h, token_ids)
                        
                    seq.block_table.append(block_id)
                    seq._last_hash = h
                    continue
            else:
                h = -1
            
            # 缓存未命中，或者不是完整的块，拿一个新的空闲块
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
            
            if is_full_block:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
                seq._last_hash = h
                
            seq.block_table.append(block_id)

        return True

    def deallocate(self, seq: Sequence):
        """释放该 Sequence 占用的所有物理块。用于请求完成或被抢占时。"""
        # 逆序释放，符合 vLLM 尾部优先驱逐的策略
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        seq.block_table.clear()
        if hasattr(seq, '_last_hash'):
            del seq._last_hash