# imu_repo/compute/backends.py
from __future__ import annotations
from typing import Any, Dict, Tuple, List
import multiprocessing as mp
from compute.registry import Backend

# ----- עזרים -----

def _vec_add_py(a: List[float], b: List[float]) -> List[float]:
    if len(a) != len(b):
        raise ValueError("vec_add_len_mismatch")
    return [a[i]+b[i] for i in range(len(a))]

def _matmul_py(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    if not a or not b: return []
    n, k = len(a), len(a[0])
    if k != len(b): raise ValueError("matmul_dim_mismatch")
    m = len(b[0])
    # תוצר n x m
    res = [[0.0]*m for _ in range(n)]
    for i in range(n):
        ai = a[i]
        for t in range(k):
            ait = ai[t]
            bt = b[t]
            for j in range(m):
                res[i][j] += ait * bt[j]
    return res

# ----- Backend CPU -----

class CPUBackend(Backend):
    name = "cpu"

    def supports(self, op: str, **shape: Any) -> bool:
        return op in ("vec_add","matmul","conv1d")

    def run(self, op: str, **kwargs: Any) -> Any:
        if op=="vec_add":
            return _vec_add_py(kwargs["a"], kwargs["b"])
        elif op=="matmul":
            return _matmul_py(kwargs["a"], kwargs["b"])
        elif op=="conv1d":
            x: List[float] = kwargs["x"]; w: List[float] = kwargs["w"]
            pad = int(kwargs.get("pad", 0)); stride = int(kwargs.get("stride",1))
            z = [0.0]*(pad)+x+[0.0]*(pad)
            out=[]
            for i in range(0, len(z)-len(w)+1, stride):
                s = 0.0
                for j in range(len(w)):
                    s += z[i+j]*w[j]
                out.append(s)
            return out
        else:
            raise RuntimeError("unknown_op")

# ----- Backend “GPU” סימולציה (ריבוי תהליכים) -----

def _mm_row(args):
    row, b = args
    m = len(b[0])
    out = [0.0]*m
    for t, aval in enumerate(row):
        bt = b[t]
        for j in range(m):
            out[j] += aval * bt[j]
    return out

class SimulatedGPUBackend(Backend):
    name = "gpu_sim"

    def __init__(self, max_workers: int | None=None):
        self.max_workers = max_workers or max(2, mp.cpu_count()//2)

    def supports(self, op: str, **shape: Any) -> bool:
        if op=="matmul":
            n = int(shape.get("n",0))
            return n >= 16   # כדאי על מטריצות גדולות
        if op=="vec_add":
            n = int(shape.get("n",0))
            return n >= 20000
        return False

    def run(self, op: str, **kwargs: Any) -> Any:
        if op=="matmul":
            a: List[List[float]] = kwargs["a"]; b: List[List[float]] = kwargs["b"]
            if not a: return []
            with mp.Pool(processes=self.max_workers) as pool:
                return pool.map(_mm_row, [(row, b) for row in a])
        elif op=="vec_add":
            a: List[float] = kwargs["a"]; b: List[float] = kwargs["b"]
            if len(a) != len(b): raise ValueError("vec_add_len_mismatch")
            chunk = max(1, len(a)//(self.max_workers*4))
            ranges = [(i, min(i+chunk, len(a))) for i in range(0, len(a), chunk)]
            def _slice_add(s,e):
                return [a[i]+b[i] for i in range(s,e)]
            with mp.Pool(processes=self.max_workers) as pool:
                parts = pool.starmap(_slice_add, ranges)
            out=[]
            for p in parts: out.extend(p)
            return out
        else:
            raise RuntimeError("unsupported_op_for_gpu_sim")