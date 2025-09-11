# server/capabilities/impl/cuda.py
from typing import Tuple, Dict, Any
from server.capabilities.types import Capability

class CUDAToolkit(Capability):
    def name(self) -> str:
        return "cuda-toolkit"

    def is_available(self) -> bool:
        return self.which("nvcc")

    def install(self) -> Tuple[bool, Dict[str, Any]]:
        # GPU toolkits need vendor installers; do not curl|sh here.
        return False, {"hint": "Install NVIDIA CUDA Toolkit; ensure 'nvcc' on PATH"}