import sys
import os

# Adds the root directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.orchestrator import DistOrchestrator

from workshop.nanovllm_hard.services.engine.llm_engine import LLMEngine
from workshop.nanovllm_hard.services.engine.scheduler import Scheduler
from workshop.nanovllm_hard.artifacts.block_mngr.headwise_block_manager import BlockManager
from workshop.nanovllm_hard.artifacts.block_mngr.query_block_manger import QueryBlockManager
from workshop.nanovllm_hard.services.model_runner.model_runner import ModelRunner
from workshop.nanovllm_hard.services import SamplingParams

def combine(**kwargs):
    orch = DistOrchestrator(world_size=kwargs.get("tensor_parallel_size", 1))
    
    engine = orch.add(LLMEngine(**kwargs))
    
    config = engine.config
        
    scheduler = orch.add(Scheduler(config))
    
    orch.register(scheduler, "add", engine)
    orch.register(scheduler, "schedule", engine)
    orch.register(scheduler, "postprocess", engine)
    orch.register(scheduler, "is_finished", engine)
    
    dist_methods_to_register = [("run", engine), ("reset", engine), ("compress", engine)]
    
    # maybe need to sync with sequence metadata. 
    
    model_runner = orch.deploy_distributed_runner(
        engine, 
        ModelRunner, 
        dist_methods=dist_methods_to_register, 
        config=config
        )
    
    block_manager = orch.add(BlockManager(config.num_kvcache_blocks, config.kvcache_block_size, config.hf_config.num_key_value_heads, config.if_fake_compress))
    
    orch.register(block_manager, "can_allocate", scheduler)
    orch.register(block_manager, "allocate", scheduler)
    orch.register(block_manager, "can_append", scheduler)
    orch.register(block_manager, "may_append", scheduler)
    orch.register(block_manager, "deallocate", scheduler)
    
    # in the unified top K selection, since all model_runner intances have same budget,
    # so we need to update the block allocation in the rank0 modelrunner 
    orch.register(block_manager, "update_blocks_post_compression", model_runner)
    orch.register(block_manager, "reset_blocks", engine)
    
    query_block_manager = orch.add(QueryBlockManager(config.max_num_seqs, config.query_window_size))

    orch.register(query_block_manager, "allocate_query", scheduler)
    orch.register(query_block_manager, "may_append_query", scheduler)
    orch.register(query_block_manager, "deallocate_query", scheduler)
    
    
    orch.finalize()
    
    return engine, SamplingParams