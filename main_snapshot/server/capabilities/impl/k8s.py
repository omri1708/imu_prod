# server/capabilities/impl/k8s.py
from typing import Tuple, Dict, Any
from server.capabilities.types import Capability

class K8sCLI(Capability):
    def name(self) -> str:
        return "k8s-cli"

    def is_available(self) -> bool:
        return self.which("kubectl")

    def install(self) -> Tuple[bool, Dict[str, Any]]:
        return False, {"hint": "Install kubectl from kubernetes.io; ensure 'kubectl' on PATH"}