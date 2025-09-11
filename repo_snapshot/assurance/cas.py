# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, hashlib, time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

class CAS:
    """
    Simple filesystem-based CAS:
      root/
        cas/
          xx/xxxxxxxx... (sha256)  -> raw bytes
        meta/
          sha256.json -> metadata (JSON)
    """
    def __init__(self, root: str = "."):
        self.root = Path(root).resolve()
        (self.root / "cas").mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(parents=True, exist_ok=True)

    def _path_for(self, digest: str) -> Path:
        sub = digest[:2]
        return self.root / "cas" / sub / digest

    def put_bytes(self, b: bytes, meta: Optional[Dict[str, Any]] = None) -> str:
        digest = sha256_bytes(b)
        p = self._path_for(digest)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with open(p, "wb") as f:
                f.write(b)
        self._write_meta(digest, meta or {})
        return digest

    def put_file(self, src: str, meta: Optional[Dict[str, Any]] = None) -> str:
        digest = sha256_file(src)
        p = self._path_for(digest)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            # hardlink if possible; else copy
            try:
                os.link(src, p)
            except Exception:
                with open(src, "rb") as s, open(p, "wb") as d:
                    d.write(s.read())
        self._write_meta(digest, meta or {"source_path": str(Path(src).resolve())})
        return digest

    def get_bytes(self, digest: str) -> bytes:
        return open(self._path_for(digest), "rb").read()

    def _write_meta(self, digest: str, meta: Dict[str, Any]):
        mpath = self.root / "meta" / f"{digest}.json"
        meta = {"_ts": time.time(), **meta}
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def meta(self, digest: str) -> Dict[str, Any]:
        mpath = self.root / "meta" / f"{digest}.json"
        if mpath.exists():
            return json.loads(mpath.read_text(encoding="utf-8"))
        return {}
