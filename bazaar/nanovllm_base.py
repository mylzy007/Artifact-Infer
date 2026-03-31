import functools
import sys
import os

# Adds the root directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.orchestrator import DistOrchestrator

from workshop.nanovllm_base.services.engine.llm_engine import LLMEngine 
from workshop.nanovllm_base.services.engine.scheduler import Scheduler
from workshop.nanovllm_base.artifacts.block_mngr.block_manager import BlockManager
from workshop.nanovllm_base.services.model_runner import ModelRunner
from workshop.nanovllm_base.services.sampling_params import SamplingParams


def combine(**kwargs):
    orch = DistOrchestrator(world_size=kwargs.get("tensor_parallel_size", 1))
    
    engine = orch.add(LLMEngine(**kwargs))
    
    config = engine.config
        
    scheduler = orch.add(Scheduler(config))
    
    orch.register(scheduler, "add", engine)
    orch.register(scheduler, "schedule", engine)
    orch.register(scheduler, "postprocess", engine)
    orch.register(scheduler, "is_finished", engine)
        
    # model_runner = orch.add(ModelRunner(config, 0, None))
    dist_methods_to_register = [("run", engine)]
    
    model_runner = orch.deploy_distributed_runner(engine, ModelRunner, dist_methods=dist_methods_to_register, config=config)
    
    # block_manager should initialized after model runner, so that the num_blocks is calculated
    block_manager = orch.add(BlockManager(config.num_kvcache_blocks, config.kvcache_block_size))
    
    orch.register(block_manager, "can_allocate", scheduler)
    orch.register(block_manager, "allocate", scheduler)
    orch.register(block_manager, "can_append", scheduler)
    orch.register(block_manager, "may_append", scheduler)
    orch.register(block_manager, "deallocate", scheduler)
    orch.register(block_manager, "reset", engine)
    
    orch.finalize()
        
    return engine, SamplingParams