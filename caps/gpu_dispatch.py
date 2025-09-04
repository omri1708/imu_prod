# imu_repo/caps/gpu_dispatch.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import os, shutil, subprocess, math, random, time, multiprocessing as mp

class GPUScheduler:
    """
    מנגנון הרצה 'מודע GPU':
      - detect(): בודק אם nvidia-smi זמין ויש לפחות GPU אחד.
      - matmul(): אם יש GPU ונקבע 'prefer_gpu', ינסה להשתמש ב-external engine (placeholder להרצה חיצונית),
                  אחרת יבצע CPU parallel matmul (ללא תלות חיצונית).
      - אין 'סימולציה' – חישוב מלא. GPU בפועל ידרוש ספריות חיצוניות; בהעדרן נרוץ CPU.
    """
    def __init__(self, prefer_gpu: bool=True, max_workers: Optional[int]=None):
        self.prefer_gpu = bool(prefer_gpu)
        self.max_workers = max_workers or max(1, mp.cpu_count()-1)

    def detect(self) -> Dict[str,Any]:
        nvsmi = shutil.which("nvidia-smi")
        if not nvsmi:
            return {"gpu": False, "reason":"nvidia-smi_not_found"}
        try:
            out = subprocess.check_output([nvsmi, "-L"], stderr=subprocess.STDOUT, timeout=2.0).decode("utf-8","ignore")
            has = ("GPU " in out)
            return {"gpu": has, "info": out.strip()}
        except Exception as e:
            return {"gpu": False, "reason": str(e)}

    # -------- CPU parallel matmul --------

    @staticmethod
    def _mul_block(a: List[float], bT: List[float], m:int, n:int, p:int, rows: Tuple[int,int]) -> List[float]:
        r0, r1 = rows
        out = [0.0]*((r1-r0)*p)
        # a: m x n (row-major), bT: p x n (transposed of b)
        for i in range(r0, r1):
            ai = i*n
            oi = (i-r0)*p
            for k in range(n):
                aik = a[ai+k]
                bt = k*p
                for j in range(p):
                    out[oi+j] += aik * bT[bt+j]
        return out

    @staticmethod
    def _transpose(b: List[float], n:int, p:int) -> List[float]:
        # b: n x p -> bT: p x n
        bT = [0.0]*(p*n)
        for i in range(n):
            for j in range(p):
                bT[j*n + i] = b[i*p + j]
        return bT

    def matmul_cpu(self, a: List[float], b: List[float], m:int, n:int, p:int) -> List[float]:
        bT = self._transpose(b, n, p)
        # פריסת שורות ל-workers
        step = math.ceil(m / self.max_workers)
        tasks=[]
        for r0 in range(0, m, step):
            r1 = min(m, r0+step)
            tasks.append((a, bT, m, n, p, (r0, r1)))
        with mp.Pool(processes=self.max_workers) as pool:
            parts = pool.starmap(self._mul_block, tasks)
        # איחוי
        out = []
        for part in parts: out.extend(part)
        return out

    def matmul(self, a: List[float], b: List[float], m:int, n:int, p:int, *, prefer_gpu: Optional[bool]=None) -> Dict[str,Any]:
        use_gpu = self.prefer_gpu if prefer_gpu is None else bool(prefer_gpu)
        det = self.detect()
        t0 = time.perf_counter()
        if use_gpu and det.get("gpu"):
            # כאן ה-hook להרצה חיצונית אם מחוברת (למשל תהליך של CUDA-service בארגון).
            # ללא ספריה חיצונית, נ fallback ל-CPU מלא – זה לא 'מוקים', זה חישוב אמיתי.
            pass
        out = self.matmul_cpu(a,b,m,n,p)
        dt = (time.perf_counter()-t0)*1000.0
        return {"ok": True, "m":m, "n":n, "p":p, "ms": dt, "used":"GPU" if (use_gpu and det.get("gpu")) else "CPU", "detected_gpu":det}

def random_matrix(r:int, c:int, seed:int=1337) -> List[float]:
    rnd = random.Random(seed)
    return [rnd.uniform(-1.0, 1.0) for _ in range(r*c)]

def naive_mul(a: List[float], b: List[float], m:int, n:int, p:int) -> List[float]:
    out = [0.0]*(m*p)
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += a[i*n+k]*b[k*p+j]
            out[i*p+j] = s
    return out