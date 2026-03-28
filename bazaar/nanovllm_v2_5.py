import functools
import sys
import os

# Adds the root directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.orchestrator import DistOrchestrator

from src.services.nanovllm_v2_5.engine.llm_engine import LLMEngine
from src.services.nanovllm_v2_5.engine.scheduler import Scheduler
from src.services.nanovllm_v2_5.engine.block_manager import BlockManager
from src.services.nanovllm_v2_5.model_runner import ModelRunner
from src.services.nanovllm_v2_5 import SamplingParams


def combine(**kwargs):
    orch = DistOrchestrator(world_size=kwargs.get("world_size", 1))
    
    engine = orch.add(LLMEngine(**kwargs))
    
    config = engine.config
        
    scheduler = orch.add(Scheduler(config))
    
    orch.register(scheduler, "add", engine)
    orch.register(scheduler, "schedule", engine)
    orch.register(scheduler, "postprocess", engine)
    orch.register(scheduler, "is_finished", engine)
        
    # model_runner = orch.add(ModelRunner(config, 0, None))
    def post_deploy_func(model_runner):
        orch.register(model_runner, "run", engine)
        
    model_runner = orch.deploy_distributed_runner(engine, ModelRunner, post_deploy_func=post_deploy_func, config=config)
    
    # block_manager should initialized after model runner, so that the num_blocks is calculated
    block_manager = orch.add(BlockManager(config.num_kvcache_blocks, config.kvcache_block_size))
    
    orch.register(block_manager, "can_allocate", scheduler)
    orch.register(block_manager, "allocate", scheduler)
    orch.register(block_manager, "can_append", scheduler)
    orch.register(block_manager, "may_append", scheduler)
    orch.register(block_manager, "deallocate", scheduler)
    
    orch.finalize()
    
    print(engine._cells["can_allocate"].origin.name)
    
    return engine, SamplingParams