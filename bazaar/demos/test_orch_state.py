import sys
import os

# Adds the root directory to the search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.orchestrator import RegistryOrchestrator
from src.core.service import BaseService
from src.core.artifact import Artifact

class MainService(BaseService):
    def __init__(self):
        super().__init__()
        
class SubService(BaseService):
    def __init__(self):
        super().__init__()

class A(Artifact):
    def __init__(self):
        super().__init__()
        self.my_port = "0000"
        # self.port = 1234

if __name__ == "__main__": 
    orch = RegistryOrchestrator()
    service_1 = orch.add(MainService())
    service_2 = orch.add(SubService())
    artifact_1 = orch.add(A())
    
    artifact_1.define_state("port", 1234)
    
    orch.connect(orch.registry[1], orch.registry[0])  # SubService -> MainService
    orch.register(artifact_1, "port", service_2)  # A.port -> MainService
    
    orch.finalize()
    
    print(artifact_1.my_port)
    artifact_1.my_port = "1111"
    print(artifact_1.my_port)
    
    print(artifact_1.port)
    print(service_2.port)
    print(service_1.port)
    
    print(id(artifact_1.port))
    print(id(service_2.port))
    print(id(service_1.port))
    
    service_1.port = 5678
    
    print(artifact_1.port)
    print(service_2.port)
    print(service_1.port)
    
    print(id(artifact_1.port))
    print(id(service_2.port))
    print(id(service_1.port))
    
    