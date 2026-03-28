import sys
import os
import atexit

# Adds the root directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.orchestrator import DistOrchestrator
from src.core.service import BaseService
from src.core.artifact import Artifact

import torch.distributed as dist
import torch

class DistrbutedArtifact(Artifact):
    def __init__(self):
        super().__init__()
        self.local_tensor = torch.tensor([0], device="cuda")
    
    def print(self):
        print(f"[Rank {dist.get_rank()}] tensor: {self.local_tensor.item()}")
    
    def increment(self, value: int = 1):
        # self.local_tensor += value
        self.local_tensor += dist.get_rank() + 1 # Just to have different values across ranks for testing
        dist.barrier()
        print(f"[Rank {dist.get_rank()}] Before all_reduce: {self.local_tensor.device}")
        dist.all_reduce(self.local_tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {dist.get_rank()} has value {self.local_tensor.item()}")

class Service(BaseService):
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    world_size = 4
    orch = DistOrchestrator(world_size)
    
    
    main_service = orch.add(Service())
    dist_a = orch.add(DistrbutedArtifact())
    
    # dist_a.define_method("increment")

    orch.register(dist_a, "increment", main_service)
    
    dist_a = orch.deploy_distributed_runner(main_service, dist_a)
    
    # print(dist_a._cells)
    print("Finalizing orchestrator...")
    orch.finalize()
    print("Orchestrator finalized. Starting distributed operations...")
    
    main_service.increment(1)
    print("Increment called. Printing values across ranks...")
    dist_a.print()
    print("Test completed. Exiting.")