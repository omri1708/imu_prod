# infra/artifact_server.py
from __future__ import annotations
import os, hashlib, time
from typing import Dict, Optional, Tuple

class ArtifactServer:
    """
    שרת ארטיפקטים מינימלי: תוכן addressable (sha256) + מטאדאטה + TTL.
    בפועל זה מאחסן בדיסק (artifacts_dir) ורושם אינדקס בזיכרון.
    """
    def __init__(self, artifacts_dir: str):
        self.dir = artifacts_dir
        os.makedirs(self.dir, exist_ok=True)
        self.idx: Dict[str, Dict] = {}  # sha -> meta

    def put_file(self, path: str, meta: Dict) -> Tuple[str, str]:
        with open(path,'rb') as f:
            b = f.read()
        h = hashlib.sha256(b).hexdigest()
        dst = os.path.join(self.dir, h)
        if not os.path.exists(dst):
            with open(dst, 'wb') as o:
                o.write(b)
        self.idx[h] = {"meta": meta, "ts": time.time(), "bytes": len(b)}
        return h, dst

    def get(self, sha: str) -> Optional[Dict]:
        return self.idx.get(sha)