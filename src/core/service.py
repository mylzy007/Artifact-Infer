# The service here covers the following scope regarding an inference engine: 
# 1. (function calls)the entrance of interfaces (engine.py, server.py)
# 2. (loops) the major component of the inference engine (scheduler.py, model_runner.py, e.t.c)

# The service is responsible for the following:
# 1. Registering artifacts to the service, and providing the interface for artifacts to call each other
# 2. Providing the interface for the upper level to call the service, 
# and hiding the implementation details of the service to the upper level

# Advanced features of the service include:
# 1. flatten all its artifacts to make up as a whole service, and provide the interface for the upper level to call the whole service,
# 2. manage the access of states registered by artifacts, 
# so that the artifacts can call each other without worrying about the state management, 
# and the service can manage the states in a unified way.

# rules for registry: 
# 1. services can have levels, but each level of service must see 
# all flattend artifacts of all lower levels of services/artifacts. 
# Artifacts' names must be in 
# so that each call of artifacts do not have to consider the recursive depth
# 2. the access of states and artifacts should follow the rules of a tree structure, 
# so that the service can manage the states in a unified way, and avoid the complexity of managing states in a graph structure.
"""
 1     5
 | \   |
 2  3  6
 |
 4
 
 each number represents a service, we have the following rules
1. A service cannot access states that is first created at level higher (including) its parents 
(e.g. 2 cannot access states created at 1)
2. A service cannot access states that does not share the same parent 
even under the same level (e.g. 2 cannot access states created at 6, but can access states created at 3)
3. A service can only access artifacts that is registered to itself and its children 
(e.g. 2 can access artifacts registered to 4, but cannot access artifacts registered to 3)
"""

import asyncio
import functools
import time
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import uuid

from src.core.artifact import Artifact

@dataclass
class Metadata: 
    artifact_id: str
    artifact_name: str 
    flattened_artifact_name: str
    is_service: bool 
    registered_artifacts: List[str] | None # list of artifacts ids
    

class BaseService(Artifact):
    def __init__(self):
        super().__init__()
        self.artifacts: dict[str, Metadata] = {} # {id: metadata} 
    
    # must be called upwards in a recursive way
    def _extend(self, other_service):
        self.artifacts.registered_artifacts.extend(other_service.registered_artifacts)
    
    def __repr__(self) -> str:
        if not self.artifacts:
            return "<BaseService: No Artifacts>"
        
        # 1. Identify "Root" nodes (those not registered as children of anyone else)
        all_children = set()
        for meta in self.artifacts.values():
            if meta.registered_artifacts:
                all_children.update(meta.registered_artifacts)
        
        roots = [id for id in self.artifacts if id not in all_children]
        
        # 2. Build the tree string
        lines = ["BaseService Structure:"]
        for i, root_id in enumerate(roots):
            is_last = (i == len(roots) - 1)
            lines.extend(self._build_tree_lines(root_id, "", is_last))
            
        return "\n".join(lines)

    def _build_tree_lines(self, artifact_id: str, prefix: str, is_last: bool) -> List[str]:
        meta = self.artifacts.get(artifact_id)
        if not meta:
            return [f"{prefix}{'└── ' if is_last else '├── '} [MISSING ID: {artifact_id}]"]

        # Choose the branch character
        marker = "└── " if is_last else "├── "
        display_name = f"{meta.artifact_name} {'(Service)' if meta.is_service else '(Artifact)'}"
        lines = [f"{prefix}{marker}{display_name}"]

        # Prepare prefix for children
        new_prefix = prefix + ("    " if is_last else "│   ")

        # Recurse if it's a service and has children
        if meta.is_service and meta.registered_artifacts:
            children_ids = meta.registered_artifacts
            for i, child_id in enumerate(children_ids):
                is_last_child = (i == len(children_ids) - 1)
                lines.extend(self._build_tree_lines(child_id, new_prefix, is_last_child))
        
        return lines
        

class AsyncBaseService(BaseService):
    """Base class using aysnc to coordinate different execution logic"""

    def __init__(self):
        super().__init__()
        self.event_loop = asyncio.get_event_loop()
    
    async def _wrap_as_async(self, func, *args, **kwargs):
        func_partial = functools.partial(func, *args, **kwargs)
        return await self.event_loop.run_in_executor(None, func_partial) 
    

