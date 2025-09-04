# imu_repo/provenance/cas.py
from __future__ import annotations
import os, json, hashlib, time, stat, threading
from typing import Optional, Dict, Any, Tuple, Iterable

class CASError(Exception): ...
class IntegrityError(CASError): ...
class NotFound(CASError): ...

def _sha256(data: bytes) -> str:
    h = hashlib.sha256(); h.update(data); return h.hexdigest()

class CAS:
    """
    תוכן־כשמו (Content-Addressable Store) פשוט על הדיסק:
      • blobs/xx/sha256  — תכולה בינארית
      • meta/xx/sha256.json — מטא־דאטה (MIME, length, kind, created_ts)
      • links/<name>.json — קיצורים שמפנים ל־sha (לוחות זמנים, 'latest', וכו')
      • audit/log.jsonl — יומן הוספות חתום (append-only; לא מבוטל)
    """
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.blobs = os.path.join(self.root, "blobs")
        self.meta  = os.path.join(self.root, "meta")
        self.links = os.path.join(self.root, "links")
        self.audit = os.path.join(self.root, "audit")
        self._lock = threading.Lock()
        for d in [self.root, self.blobs, self.meta, self.links, self.audit]:
            os.makedirs(d, exist_ok=True)
        # הפוך את הלוג לקריא־לכולם ו־append-only ברמת API
        self.log_path = os.path.join(self.audit, "log.jsonl")

    def put(self, data: bytes, *, kind: str, mime: str="application/octet-stream",
            extra_meta: Optional[Dict[str,Any]]=None) -> str:
        sha = _sha256(data)
        sub = os.path.join(self.blobs, sha[:2])
        os.makedirs(sub, exist_ok=True)
        blob_path = os.path.join(sub, sha)
        if not os.path.exists(blob_path):
            with open(blob_path, "wb") as f:
                f.write(data)
        meta_dir = os.path.join(self.meta, sha[:2])
        os.makedirs(meta_dir, exist_ok=True)
        meta_path = os.path.join(meta_dir, f"{sha}.json")
        if not os.path.exists(meta_path):
            meta = {
                "sha256": sha, "len": len(data), "mime": mime, "kind": kind,
                "created_ts": int(time.time()),
            }
            if extra_meta: meta.update(extra_meta)
            tmp = meta_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            os.replace(tmp, meta_path)
        self._append_audit({"op":"put","sha256":sha,"kind":kind,"len":len(data)})
        return sha

    def get(self, sha: str) -> bytes:
        blob_path = os.path.join(self.blobs, sha[:2], sha)
        if not os.path.exists(blob_path): raise NotFound(sha)
        with open(blob_path, "rb") as f: return f.read()

    def meta_of(self, sha: str) -> Dict[str,Any]:
        meta_path = os.path.join(self.meta, sha[:2], f"{sha}.json")
        if not os.path.exists(meta_path): raise NotFound(sha)
        with open(meta_path, "r", encoding="utf-8") as f: return json.load(f)

    def link(self, name: str, sha: str, *, note: str="") -> None:
        self.meta_of(sha)  # validate exists
        path = os.path.join(self.links, f"{name}.json")
        payload = {"name": name, "sha256": sha, "ts": int(time.time()), "note": note}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        self._append_audit({"op":"link","name":name,"sha256":sha,"note":note})

    def resolve(self, name: str) -> Dict[str,Any]:
        path = os.path.join(self.links, f"{name}.json")
        if not os.path.exists(path): raise NotFound(name)
        with open(path, "r", encoding="utf-8") as f: return json.load(f)

    def verify_blob(self, sha: str) -> None:
        data = self.get(sha)
        calc = _sha256(data)
        if calc != sha: raise IntegrityError(f"mismatch for {sha}")

    def _append_audit(self, entry: Dict[str,Any]) -> None:
        entry = dict(entry); entry["t"] = int(time.time())
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def iter_audit(self) -> Iterable[Dict[str,Any]]:
        if not os.path.exists(self.log_path): return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: yield json.loads(line)
                except Exception: continue