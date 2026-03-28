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
        self.hello = "hello, my name is main service"
    
    def another_print(self):
        print(self.hello)
        
class SubService(BaseService):
    def __init__(self):
        super().__init__()
        self.hello = "hello, my name is sub service"
    
    def another_print(self):
        print(self.hello)

class A(Artifact):
    def __init__(self):
        super().__init__()
    
    def print_port(self):
        print(f"Port is: {self.port}")
    
    def my_print(self):
        print("Hello from A.my_print!")

    def another_print(self):
        print(self.hello + "from A's another_print()")
    
if __name__ == "__main__": 
    orch = RegistryOrchestrator()
    service_1 = orch.add(MainService())
    service_2 = orch.add(SubService())
    artifact_1 = orch.add(A())
    
    artifact_1.define_state("port", 1234)
    
    artifact_1.my_print()  # Direct call to A's method
    
    orch.connect(orch.registry[1], orch.registry[0])  # SubService -> MainService
    orch.register(artifact_1, "port", service_2)  # A.port -> MainService
    
    orch.register(artifact_1, "print_port", service_2)  # A.print_port -> A
    
    orch.register(artifact_1, "another_print", service_2)  # A.another_print -> A
    
    orch.finalize()
            
    service_1.port = 5678
    
    artifact_1.print_port()
    service_1.print_port()
    service_2.print_port()
    
    # NOTE in the current version the child's mehtod will not override the parent's method
    # we may need another interface like Arifact.patch() to explicitly allow this in the future
    
    service_1.another_print()
    service_2.another_print()
    