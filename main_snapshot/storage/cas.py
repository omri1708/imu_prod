# storage/cas.py (Content-Addressable Store + evidence)
# -*- coding: utf-8 -*-
import os, hashlib, json
from typing import Tuple

_BASE = "var/cas"
os.makedirs(_BASE, exist_ok=True)

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def put_bytes(data: bytes) -> Tuple[str,str]:
    h = _sha256(data)
    p = os.path.join(_BASE, h[:2], h[2:])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(data)
    return h, p

def put_text(txt: str, *, meta: dict = None) -> Tuple[str,str]:
    h, p = put_bytes(txt.encode("utf-8"))
    if meta:
        with open(p + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return h, p

def exists(hexhash: str) -> bool:
    p = os.path.join(_BASE, hexhash[:2], hexhash[2:])
    return os.path.exists(p)

def path(hexhash: str) -> str:
    return os.path.join(_BASE, hexhash[:2], hexhash[2:])