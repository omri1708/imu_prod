# imu_repo/compute/ops.py
from __future__ import annotations
from typing import List, Any, Dict
from compute.registry import REGISTRY
from compute.backends import CPUBackend, SimulatedGPUBackend

# רישום ברירת מחדל (CPU + “GPU” סימולציה)
if not any(isinstance(b, CPUBackend) for b in REGISTRY.backends):
    REGISTRY.register(CPUBackend())
if not any(isinstance(b, SimulatedGPUBackend) for b in REGISTRY.backends):
    REGISTRY.register(SimulatedGPUBackend())

def vec_add(a: List[float], b: List[float]) -> List[float]:
    return REGISTRY.run("vec_add", a=a, b=b, _shape={"n": len(a)})

def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a); k = len(a[0]) if a else 0; m=len(b[0]) if b else 0
    return REGISTRY.run("matmul", a=a, b=b, _shape={"n":n,"k":k,"m":m})

def conv1d(x: List[float], w: List[float], *, pad: int=0, stride: int=1) -> List[float]:
    return REGISTRY.run("conv1d", x=x, w=w, _shape={"n": len(x), "kw": len(w), "pad": pad, "stride": stride})