# This version of block manager does not implement with prefix caching
# only support block_size == 1 in this version as well
from collections import deque
import xxhash
import numpy as np

from workshop.nanovllm_hard.services.engine.sequence import Sequence, torch_rotl_uint8

from src.core.service import BaseService

import torch
import torch.distributed as dist

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.token_ids = []

    def update(self, token_ids: list[int]):
        self.token_ids = token_ids

    def reset(self):
        self.token_ids = []


class BlockManager(BaseService):
    num_kv_heads = 8
    @property
    def name(self):
        return "HeadwiseBlockManager"

    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int, if_fake_compress=True, *args, **kwargs):
        super().__init__()
        assert num_blocks > 0
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        # this is different from config.if_fake_compress, should be decrecated
        self.if_fake_compress = if_fake_compress
        # self.if_fake_compress = True
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # during the engine running, some block will be released head by head, when all heads are released, the block can be reused
        self.released_block_ids: torch.Tensor = torch.zeros((num_blocks,), dtype=torch.uint8, device="cpu")

    def reset_blocks(self):
        num_blocks = len(self.blocks)
        [Block(i) for i in range(num_blocks)]
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()
    
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        block.reset()
        self.free_block_ids.remove(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        self.free_block_ids.append(block_id)
        self.released_block_ids[block_id] = 0
        
    def update_blocks_post_compression(self, seq: Sequence, budget: int):
        for block_id in reversed(seq.block_table[budget:]):
            self._deallocate_block(block_id)
        seq.block_table = seq.block_table[:budget]  
        
        # NOTE need further design
        seq.headwise_mask_layer_transpose = seq.headwise_mask_layer_transpose[..., :(seq.num_blocks_max_heads + 7) // 8]
    
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks_max_heads

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        for i in range(seq.num_blocks_max_heads):
            token_ids = seq.block(i)
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
            block.update(token_ids)
            seq.block_table.append(block_id)
            seq.block_id_to_count[block_id] = 0
            
            # if i % 8 == 0 and i != 0:
            #     seq.headwise_mask_layer_transpose = torch.cat(
            #         [seq.headwise_mask_layer_transpose, torch.zeros((seq.num_layers, seq.num_kv_heads, 1), device="cuda", dtype=torch.uint8)], dim=2
            #     )
            # seq.headwise_mask_layer_transpose[:, :, i // 8] += seq.next_mask.to(seq.headwise_mask_layer_transpose.device)
            # seq.next_mask = torch_rotl_uint8(seq.next_mask, 1)
            
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self._deallocate_block(block_id)
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # print([self.blocks[index].hash for index in block_table]) 
        # NOTE when the block == 1, the handling logic is different 
        assert self.block_size == 1
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        seq.block_table.append(block_id)
        seq.block_id_to_count[block_id] = 0
