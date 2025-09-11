# common/provenance.py
from __future__ import annotations
import os, json, hashlib, time
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class Evidence:
    sha256: str
    path: str
    created_ts: float
    trust: Literal["low","medium","high"] = "medium"
    signer: Optional[str]=None
    sig_algo: Optional[str]=None
    signature: Optional[str]=None

class CAS:
    def __init__(self, root="cas"):
        self.root=root; os.makedirs(root, exist_ok=True)
    def _blob_path(self, digest:str)->str:
        return os.path.join(self.root, digest[:2], digest[2:])
    def put(self, path:str)->Evidence:
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for chunk in iter(lambda:f.read(1<<20), b''):
                h.update(chunk)
        digest=h.hexdigest()
        dst=self._blob_path(digest)
        os.makedirs(os.path.dirname(dst),exist_ok=True)
        if not os.path.exists(dst):
            # hard-link or copy
            try: os.link(path, dst)
            except: 
                import shutil; shutil.copy2(path,dst)
        ev=Evidence(sha256=digest, path=os.path.abspath(path), created_ts=time.time())
        self._write_meta(ev)
        return ev
    def _write_meta(self, ev:Evidence):
        meta=os.path.join(self.root,"meta",ev.sha256+".json")
        os.makedirs(os.path.dirname(meta),exist_ok=True)
        with open(meta,"w",encoding="utf-8") as f:
            json.dump(ev.__dict__, f, ensure_ascii=False, indent=2)