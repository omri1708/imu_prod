# imu_repo/gpu/runtime.py
from __future__ import annotations
from typing import List

class ResourceRequired(Exception): ...

def matmul_cpu(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a); m = len(a[0]); p = len(b[0])
    # basic dimension check
    if m != len(b):
        raise ValueError("incompatible_dimensions")
    out = [[0.0]*p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            aik = a[i][k]
            for j in range(p):
                out[i][j] += aik * b[k][j]
    return out

def matmul(a: List[List[float]], b: List[List[float]], prefer_gpu: bool = False) -> List[List[float]]:
    """
    Matrix multiplication with optional GPU.
    - If prefer_gpu and no GPU runtime, raises ResourceRequired.
    - Otherwise falls back to CPU.
    """
    if not prefer_gpu:
        return matmul_cpu(a,b)
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as drv
        import pycuda.gpuarray as gpuarray
        import numpy as np
        from skcuda import linalg as culinalg
    except Exception as e:
        raise ResourceRequired("GPU stack (pycuda + scikit-cuda + numpy) required") from e

    import numpy as np
    import pycuda.gpuarray as gpuarray
    from skcuda import linalg as culinalg
    culinalg.init()

    A = np.array(a, dtype=np.float32)
    B = np.array(b, dtype=np.float32)
    dA = gpuarray.to_gpu(A)
    dB = gpuarray.to_gpu(B)
    dC = culinalg.dot(dA, dB)
    C = dC.get()
    return C.tolist()
