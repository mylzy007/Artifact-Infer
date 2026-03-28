# The artifact here serves as either "stateless" or "independently stateful components" 
# that can be registered to the service, and can call each other through the service.
# An artifact can be a service itself, so that it can register other artifacts to itself, 
# and provide the interface for other artifacts to call each other.
# The artifact is responsible for the following:
# 1. Providing the interface for other artifacts to call each other, and hiding the implementation details of the artifact to other artifacts
# 2. Providing the interface for the service to call the artifact, and hiding the implementation details of the artifact to the service

"""
Base interfaces for artifacts

Provides the fundamental building blocks for systematic implementations
that can be registered, called, and composed within the unified service architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
from pathlib import Path
import importlib.util
import inspect
import uuid

from typing import Any, Dict, Optional
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import pickle

import torch
import torch.distributed as dist

# @dataclass
# class ExecutionContext:
#     """Context passed to artifacts during execution"""

#     artifact_name: str
#     metadata: Dict[str, Any]
#     request_id: Optional[str] = None


# """TODO let's see if there are better abstraction for the 'runnning' of artifacts"""
# @abstractmethod
# def execute(self, input_data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
#     """
#     Single unified API for artifact execution

#     Args:
#         input_data: Dictionary containing all inputs
#         context: Execution context with mode and metadata

#     Returns:
#         Dictionary with execution results
#     """

class Artifact:
    def __init__(self):
        super().__init__()
        self.parents: List['Artifact'] = []
        self._cells: Dict[str, Cell] = {}
        self._state_map: Dict[str, str] = {}
    
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def define_state(self, name: str, value: Any = None):
        if hasattr(self, name):
            self._cells[name] = StateCell(getattr(self, name)) if value is None else StateCell(value)
        else:
            self._cells[name] = StateCell(value)

    def define_method(self, name: str):
        """Wraps a class method into a MethodCell."""
        print(f"Defining method '{name}' for artifact '{self.name}' from origin '{self.__class__.__name__}'")
        func = getattr(self.__class__, name)
        self._cells[name] = MethodCell(func, self)
    
    # def __getattribute__(self, name: str) -> Any:
    #     # Avoid infinite recursion by using super() to get '_cells'
    #     # We use __dict__.get to avoid triggering this very method recursively
    #     d = super().__getattribute__('__dict__')
    #     cells = d.get('_cells', {})

    #     # PRIORITY 1: Check if it's a registered Cell
    #     if name in cells:
    #         return cells[name].get_value(self)

    #     # PRIORITY 2: Fall back to standard Python behavior
    #     return super().__getattribute__(name)

    def __getattr__(self, name: str) -> Any:
        d = self.__dict__
        cells = d.get('_cells')
        # print(f"__getattr__ triggered for '{name}' on '{self.name}' with cells: {list(cells.keys()) if cells else None}")
        # 1. direct call by local_alias (i.e. name will always be the local_alias)
        if cells and name in cells:
            # We pass 'self' so MethodCells can build the correct Proxy
            return cells[name].get_value(self)
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If nn.Module can't find it either, raise a standard error
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any):
        cells = self.__dict__.get('_cells')
        if cells and name in cells:
            cell = cells[name]
            if isinstance(cell, StateCell):
                cell.value = value
            else:
                raise AttributeError(f"Cannot rebind MethodCell '{name}'")
        else:
            super().__setattr__(name, value)
    
    # @abstractmethod
    # def get_schema(self) -> Dict[str, Any]:
    #     """Return JSON schema for input validation"""
    #     pass

    # @abstractmethod
    # def get_description(self) -> str:
    #     """Human-readable description"""
    #     return self.__class__.__doc__ or "No description provided"

    # @abstractmethod
    # def validate_input(self, input_data: Dict[str, Any]) -> bool:
    #     """Validate input against schema (optional override)"""
    #     return True  # Basic validation - can be overridden

class Cell:
    """Base interface for a registry container."""
    def get_value(self, host: Artifact) -> Any:
        raise NotImplementedError
    
    def serialize(self) -> Dict[str, Any]:
        """Pack the cell into a purely picklable dictionary."""
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: Dict[str, Any], host: 'Artifact') -> 'Cell':
        """Reconstruct the cell inside the worker process."""
        raise NotImplementedError

class StateCell(Cell):
    def __init__(self, initial_value: Any = None):
        self.value = initial_value

    def get_value(self, host: Artifact) -> Any:
        return self.value
    
    def serialize(self) -> Dict[str, Any]:
        return {"type": "StateCell", "value": self.value}

    @classmethod
    def deserialize(cls, data: Dict[str, Any], host: 'Artifact') -> 'Cell':
        return cls(initial_value=data["value"])

class MethodCell(Cell):
    def __init__(self, func: Callable, origin: Artifact):
        self.func = func
        self.origin = origin
        print(f"Created MethodCell for '{func.__name__}' from origin '{origin.name}'")

    def get_value(self, host: Artifact) -> Any:
        # Returns a wrapper that injects the MethodProxy at call-time
        def wrapper(*args, **kwargs):
            proxy = MethodProxy(self.origin, host)
            return self.func(proxy, *args, **kwargs)
        return wrapper
    
    def serialize(self) -> Dict[str, Any]:
        # We cannot pickle functions reliably, so we send the name!
        return {
            "type": "MethodCell", 
            "func_name": self.func.__name__,
            "origin_name": self.origin.name
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any], host: 'Artifact') -> 'Cell':
        # The worker dynamically grabs its own local memory function
        func = getattr(host.__class__, data["func_name"])
        return cls(func, host)
    
class DistMethodCell(Cell):
    def __init__(self, func, origin, rank, world_size, ipc_wrapper: "IPCWorkerWrapper"):
        self.func = func
        self.origin = origin
        self.rank = rank
        self.world_size = world_size
        self.ipc_wrapper = ipc_wrapper # Reference to the unified handler

    def get_value(self, host):
        def dist_wrapper(*args, **kwargs):
            if self.rank == 0 and self.world_size > 1:
                # Delegate the broadcast logic to the unified wrapper
                self.ipc_wrapper.write_shm(self.func.__name__, *args)
            
            # Execute locally for Rank 0
            proxy = MethodProxy(self.origin, host)
            return self.func(proxy, *args, **kwargs)
            
        return dist_wrapper
    
    def serialize(self) -> Dict[str, Any]:
        return {
            "type": "DistMethodCell",
            "func_name": self.func.__name__,
            "origin_name": self.origin.name
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any], host: 'Artifact') -> 'Cell':
        # Fallback to standard MethodCell for the workers!
        func = getattr(host.__class__, data["func_name"])
        return MethodCell(func, host)

class MethodProxy:
    def __init__(self, origin, host):
        self.__dict__['_origin'] = origin
        self.__dict__['_host'] = host

    def __getattr__(self, name):
        if hasattr(self._origin, name): return getattr(self._origin, name)
        if hasattr(self._host, name): return getattr(self._host, name)
        raise AttributeError(f"Proxy cannot resolve '{name}' from either origin '{self._origin.name}' or host '{self._host.name}'")

    def __setattr__(self, name, value):
        if hasattr(self._origin, name): setattr(self._origin, name, value)
        else: setattr(self._origin, name, value)


class IPCWorkerWrapper():
    """Handles all IPC read/write protocols for both Master and Workers."""
    def __init__(self, proc: 'Artifact', rank: int, world_size: int, events: Event | list[Event], shm: SharedMemory):
        self.proc = proc
        self.rank = rank
        self.world_size = world_size
        self.events = events # List for Rank 0, single Event for Ranks 1-N
        self.shm = shm
    
    def _set_dist_env(self, world_size, rank):
        print("before init dist proc group")
        dist.init_process_group(backend="nccl", init_method="tcp://localhost:54321", world_size=world_size, rank=rank)
        print("after init dist proc group")
        torch.cuda.set_device(rank)
        self.device = torch.device("cuda", rank)
        torch.set_default_device("cuda")
    
    @classmethod 
    def create_from_cls(cls, proc_cls, rank, world_size, rank_event, shm, **kwargs):
        cls._set_dist_env(cls, world_size, rank)
        subproc = proc_cls(**kwargs)
        return cls(subproc, rank, world_size, rank_event, shm)

    @classmethod
    def create_master(cls, proc_cls, world_size, events, shm, post_deploy_func=None, **kwargs):
        cls._set_dist_env(cls, world_size, rank=0)
        proc = proc_cls(**kwargs)
        if post_deploy_func:
            post_deploy_func(proc)
        return cls(proc, rank=0, world_size=world_size, events=events, shm=shm)
    
    def sync_registry(self):
        """Called by Rank 0 to broadcast finalized DAG state."""
        if not (self.world_size > 1 and self.rank == 0):
            return 
        
        serialized_cells = {}
        for name, cell in self.proc._cells.items():
            serialized_cells[name] = cell.serialize()
            
        payload = {
            "_state_map": self.proc._state_map,
            "_cells": serialized_cells
        }
        self.write_shm("__SYNC_REGISTRY__", payload)
        
        dist.barrier()
    
    def exit(self):
        if self.world_size > 1:
            if self.rank == 0:
                self.write_shm("exit")
                self.shm.unlink()
            self.shm.close()
            dist.barrier()
            
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.events.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, args = pickle.loads(self.shm.buf[4 : n + 4])
        self.events.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.events:
            event.set()

    def loop(self):
        """Worker loop (Ranks 1-N only)."""
        while True:
            method_name, args = self.read_shm()
            if method_name == "exit":
                self.exit()
                break
            elif method_name == "__SYNC_REGISTRY__":
                payload = args[0]
                
                # 1. Sync the Alias Map
                self.proc._state_map.update(payload["_state_map"])
                
                # 2. Rebuild the Cells Polymorphically
                for name, cell_data in payload["_cells"].items():
                    cell_type = cell_data["type"]
                    
                    if cell_type == "StateCell":
                        self.proc._cells[name] = StateCell.deserialize(cell_data, self.proc)
                    elif cell_type in ["MethodCell", "DistMethodCell"]:
                        self.proc._cells[name] = DistMethodCell.deserialize(cell_data, self.proc)

                    dist.barrier()
            else:
                method = getattr(self.proc, method_name)
                method(*args)

class ArtifactRegistry:
    pass
