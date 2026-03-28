from typing import Any, Dict, List

from .artifact import (Artifact, 
                       Cell, 
                       DistMethodCell, 
                       DistMethodCell, 
                       MethodCell, 
                       IPCWorkerWrapper)
from .service import BaseService
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

import pickle
import torch
import torch.distributed as dist

import atexit

from src.core import artifact

class RegistryOrchestrator:
    def __init__(self):
        self.registry: List[Artifact] = []

    def add(self, artifact: Artifact):
        if artifact not in self.registry:
            self.registry.append(artifact)
        return artifact
    
    def register(self, child: Artifact, attr_name: str, parent: Artifact):
        """Unified entry point for both states and methods."""
        # Ensure it exists in the artifact's cells
        if attr_name not in child._cells:
            # Auto-wrap if it's a method not yet defined as a cell
            attr = getattr(child, attr_name)
            if callable(attr):
                child.define_method(attr_name)
            else:
                child.define_state(attr_name, attr)
        
        if child not in self.registry:
            self.registry.append(child)
        
        if parent not in child.parents:
            child.parents.append(parent)
        
    def connect(self, child: Artifact, parent: Artifact):
        """Phase 1: Build the Topology."""
        if parent not in child.parents:
            child.parents.append(parent)

    def finalize(self):
        """Phase 2: Validate and Bind."""
        # 1. Topology Validation
        self._check_cycles()
        
        # 2. Propagation
        # We propagate every cell defined in every artifact to all its ancestors
        for artifact in self.registry:
            # We only propagate cells that were "Defined" on this specific artifact
            # (i.e., not the ones it inherited)
            for attr_name, cell in artifact._cells.items():
                # We skip inherited cells during this primary loop to avoid redundant paths
                if isinstance(cell, MethodCell) and cell.origin != artifact:
                    continue
                origin_alias = f"{artifact.name}_{attr_name}"
                for parent in artifact.parents:
                    self._propagate(parent, attr_name, origin_alias, cell)

    def _propagate(self, service: Artifact, local_alias: str, origin_alias: str, cell: Cell):
        # Bind reference
        print(f"Propagating {local_alias} from {cell.origin.name} to {service.name} as {origin_alias} with cell: {id(cell)}")
        service._cells[local_alias] = cell
        service._state_map[local_alias] = origin_alias
        
        # Climb the DAG
        for parent in service.parents:
            self._propagate(parent, local_alias, origin_alias, cell)

    def _check_cycles(self):
        visited, stack = set(), set()
        def dfs(node):
            visited.add(node.name)
            stack.add(node.name)
            for p in node.parents:
                if p.name not in visited:
                    dfs(p)
                elif p.name in stack:
                    raise RecursionError(f"Cycle detected: {node.name} -> {p.name}")
            stack.remove(node.name)
        for art in self.registry:
            if art.name not in visited: dfs(art)
        
def _boostrap(subproc_cls, rank, world_size, rank_event, **kwargs):
    print(f"Child process for Rank {rank} initializing.")
    shm = SharedMemory(name="nanovllm")
    wrapped_instance = IPCWorkerWrapper.create_from_cls(subproc_cls, rank, world_size, rank_event, shm, **kwargs)
    wrapped_instance.loop()

class DistOrchestrator(RegistryOrchestrator):
    def __init__(self, world_size: int, shm_size: int = 2**20):
        super().__init__()
        self.world_size = world_size
        self.ctx = mp.get_context("spawn")
        self.processes = []
        
        self.rank0_wrappers = []
        # Setup IPC primitives globally for this TP group
        if self.world_size > 1:
            self.shm = SharedMemory(name="nanovllm", create=True, size=shm_size)
            self.events = [self.ctx.Event() for _ in range(1, world_size)]
        else:
            self.shm, self.events = None, []
    
    def exit(self):
        self.wrapped_rank0_child.exit()
        for p in self.processes:
            p.join()
    
    def deploy_distributed_runner(self, parent: Artifact, child_cls: type, post_deploy_func=None, **kwargs) -> Artifact:
        print("[Reminder] You should always register the states/methods as MethodCell before deploy distributed processes")
        """
        Deploys the distributed group, links Rank 0 to the parent, and returns Rank 0.
        """
        
        if self.world_size > 1:
            for i in range(1, self.world_size):
                rank_event = self.events[i - 1]
                
                p = self.ctx.Process(
                    target=_boostrap, 
                    args=(child_cls, i, self.world_size, rank_event),
                    kwargs=kwargs 
                )
                p.start()
                self.processes.append(p)
        
        self.wrapped_rank0_child = IPCWorkerWrapper.create_master(child_cls, self.world_size, self.events, self.shm, post_deploy_func, **kwargs)

        rank0_child = self.wrapped_rank0_child.proc
        
        self.rank0_wrappers.append(self.wrapped_rank0_child)
        
        self.connect(child=rank0_child, parent=parent)
        
        # Wrap methods in the Distributed Broadcast Cell
        for name, cell in list(rank0_child._cells.items()):
            if isinstance(cell, MethodCell):
                rank0_child._cells[name] = DistMethodCell(
                    func=cell.func, 
                    origin=cell.origin, 
                    rank=0, 
                    world_size=self.world_size,
                    ipc_wrapper=self.wrapped_rank0_child # The cell uses the wrapper for write_shm
                )
        
        atexit.register(self.exit)
        # Return Rank 0 for manual pre-finalization hooks
        return rank0_child
    
    def finalize(self):
        """Phase 2: Build DAG and Sync Workers."""
        # 1. Standard DAG Propagation in the main process
        super().finalize()
        
        # 2. Late-Sync! 
        # Rank 0 now has all the propagated states from LLMEngine. 
        # Tell the wrappers to push this data to the waiting workers.
        for wrapper in self.rank0_wrappers:
            wrapper.sync_registry()
        
