# imu_repo/user/semvec.py
from __future__ import annotations
import re, math, hashlib
from typing import Dict, List, Tuple

TOKEN = re.compile(r"[A-Za-zא-ת0-9]+", re.U)


def _ngrams(tok: str, n: int = 3) -> List[str]:
    s=f"^{tok}$"
    return [s[i:i+n] for i in range(max(1,len(s)-n+1))]


def _h(s: str, buckets: int = 2048) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(),16) % buckets


def embed(text: str, buckets: int = 2048) -> List[float]:
    vec = [0.0]*buckets
    toks = [t.lower() for t in TOKEN.findall(text)]
    if not toks: return vec
    for t in toks:
        for g in _ngrams(t,3):
            vec[_h(g,buckets)] += 1.0
    # normalize
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v/norm for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x*y for x,y in zip(a,b))