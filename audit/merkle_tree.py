from __future__ import annotations
import os, hashlib
from typing import Dict, Any, Tuple, List

def _h(b: bytes) -> str: return hashlib.sha256(b).hexdigest()

def hash_file(path: str) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

def merkle_dir(root: str) -> Tuple[str, Dict[str,Any]]:
    leaves=[]
    for dp,_,fs in os.walk(root):
        for fn in sorted(fs):
            p=os.path.join(dp,fn); leaves.append((_h(p.encode()), hash_file(p), os.path.relpath(p, root)))
    # בניית עץ פשטני: שרשור זוגות
    level=[_h((a+b+name).encode()) for (a,b,name) in leaves] or ["0"*64]
    nodes={"leaves":[{"path":name,"ph":a,"fh":b} for (a,b,name) in leaves], "levels":[level]}
    while len(level)>1:
        nxt=[]
        for i in range(0,len(level),2):
            x=level[i]; y=level[i+1] if i+1<len(level) else level[i]
            nxt.append(_h((x+y).encode()))
        nodes["levels"].append(nxt)
        level=nxt
    root_hash=level[0]
    return root_hash, nodes
