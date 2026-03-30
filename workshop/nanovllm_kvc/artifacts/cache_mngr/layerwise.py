from src.core.service import BaseService

from ..attention_backend.flashinfer_attention import (
    store_kvcache,
    read_kvcache,
    read_q_cache,
)
import torch

import itertools

from workshop.nanovllm_kvc.services.utils.logging import get_log, set_log

class CacheManager(BaseService):
    @property
    def name(self):
        return "CacheManagerLayerwise"

    """
    This version of implementation only 
    """

    def __init__(self, config, compressor=None):
        super().__init__()
        # NOTE
        # attention_backend.register(self)

        self.num_layers = config.hf_config.num_hidden_layers
        
        self.budget = config.layer_budget

        self.seq_to_layer_block_table = {}
        
        self.cu_page_indices = self.cu_seq_lens = None

        # NOTE
        self.compressor = compressor
        
        self.cu_seqs = []
    

    def log_page_indices(self, seqs):
        # move to model runner before capturing cuda graph
        self.cu_seqs = seqs
        occupied_pages = 0
        cu_page_indices = torch.tensor(
            list(itertools.chain(*[seq.block_table for seq in seqs]))
        ).to(torch.int32)
        occupied_pages = cu_page_indices.shape[0]
        seq_lens = torch.tensor(
            [len(seq.block_table) for seq in self.cu_seqs]
        ).to(torch.int32)
        self.page_indices = cu_page_indices
        self.seq_lens = seq_lens
        log = get_log()
        log.occupied_pages = occupied_pages
        set_log(log)
        
    def update_indices(self):
        self.prepare_metadata_for_attn(
            self.seq_lens,
            self.page_indices, 
        )

    def update_indices_capture(self, bs: int):
        self.init_forward_metadata_capture_cuda_graph(
            bs,
            self.seq_lens,
            self.page_indices,
        )

    def update_indices_replay(self, bs: int):
        self.init_forward_metadata_replay_cuda_graph(
            bs,
            self.seq_lens,
            self.page_indices, 
        )
    
    # NOTE should be refactored or deprecated
    def read_and_store_cache_iterative(self, q_cache, k_cache, v_cache, layer_id):
        total_steps = (len(self.cu_seqs[0].block_table) - self.budget) // self.config.steps_between_cache_compressions + 1
        for i in range(total_steps):
            start = i * self.config.steps_between_cache_compressions + self.budget
            end = min(start + self.config.steps_between_cache_compressions, len(self.cu_seqs[0].block_table))
            slot_mappings = self.cu_seqs[0].block_table[:self.budget] + self.cu_seqs[0].block_table[start:end]
            assert len(self.cu_seqs) == 1, "Currently only support single request"

            slot_mappings_tensor = torch.tensor(slot_mappings, device="cuda").to(
                torch.int32
            )
            
            query_slot_mapping = [self.cu_seqs[0].query_block_id]

            query_slot_mapping_tensor = torch.tensor(query_slot_mapping, device="cuda").to(
                torch.int32
            )

            query = read_q_cache(
                q_cache=q_cache,
                query_slot_mapping=query_slot_mapping_tensor,
            )

            key, value = read_kvcache(
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=slot_mappings_tensor,
            )
            
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

            # NOTE
            updated_k, updated_v = self.compressor.update_kv(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                self.cu_seqs[0],
                layer_id, 
            )

            key = updated_k.transpose(1, 2).squeeze(0).contiguous()
            value = updated_v.transpose(1, 2).squeeze(0).contiguous()

            slot_mappings_tensor = slot_mappings_tensor[: key.shape[0]]
            
            store_kvcache(
                key=key,
                value=value,
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=slot_mappings_tensor,
            )
    
    def read_and_store_cache(self, q_cache, k_cache, v_cache, layer_id):
        """
        option 1: per-sequence handling

        option 2: like flashinfer's layout, handling with packed indices,
        """
        
        for seq in self.cu_seqs:
        
            slot_mappings = seq.block_table

            # assert len(self.cu_seqs) == 1, "Currently only support single request"

            slot_mappings_tensor = torch.tensor(slot_mappings, device="cuda").to(
                torch.int32
            )
            
            query_slot_mapping = [seq.query_block_id]

            query_slot_mapping_tensor = torch.tensor(query_slot_mapping, device="cuda").to(
                torch.int32
            )

            query = read_q_cache(
                q_cache=q_cache,
                query_slot_mapping=query_slot_mapping_tensor,
            )

            key, value = read_kvcache(
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=slot_mappings_tensor,
            )
            
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)

            # NOTE
            ret = self.compressor.update_kv(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                self.cu_seqs[0],
                layer_id, 
            )
            
            updated_k = ret["key_states"]
            updated_v = ret["value_states"]

            key = updated_k.transpose(1, 2).squeeze(0).contiguous()
            value = updated_v.transpose(1, 2).squeeze(0).contiguous()
            
            slot_mappings_tensor = slot_mappings_tensor[: key.shape[0]]
        
            store_kvcache(
                key=key,
                value=value,
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=slot_mappings_tensor,
            )
