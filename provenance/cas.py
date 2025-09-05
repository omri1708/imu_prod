# imu_repo/provenance/cas.py
from __future__ import annotations
import os, json, hashlib, time, stat, threading
from typing import Optional, Dict, Any, Tuple, Iterable
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict
from policy.user_policies import ttl_for

class CASError(Exception): ...
class IntegrityError(CASError): ...
class NotFound(CASError): ...


CAS_ROOT = os.environ.get("IMU_CAS_ROOT", ".imu_cas")


os.makedirs(CAS_ROOT, exist_ok=True)


Trust = Literal["low","medium","high","system"]

@dataclass
class CASMeta:
    kind: str           # "evidence" | "artifact" | "ui" | "log" | ...
    user_id: str
    trust: Trust
    created_ts: float
    ttl_seconds: int
    source_url: Optional[str] = None
    signature: Optional[str] = None     # מקום לחתימה דיגיטלית
    note: Optional[str] = None

class ContentAddressableStore:
    def __init__(self, root:str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, digest:str) -> str:
        return os.path.join(self.root, digest[0:2], digest[2:4], digest)

    def put_bytes(self, b:bytes, meta:CASMeta) -> str:
        digest = hashlib.sha256(b).hexdigest()
        p = self._path(digest)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p):
            with open(p, "wb") as f: f.write(b)
        with open(p+".meta.json","w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, ensure_ascii=False, indent=2)
        return digest

    def get_bytes(self, digest:str) -> bytes:
        with open(self._path(digest), "rb") as f:
            return f.read()

    def get_meta(self, digest:str) -> CASMeta:
        with open(self._path(digest)+".meta.json","r", encoding="utf-8") as f:
            d = json.load(f)
        return CASMeta(**d)

    def gc(self, now:Optional[float]=None) -> int:
        """מוחק קבצים שפג תוקפם לפי ה-TTL של המדיניות בעת ההפקדה."""
        now = now or time.time()
        removed = 0
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for fn in filenames:
                if not fn.endswith(".meta.json"): 
                    continue
                meta_path = os.path.join(dirpath, fn)
                with open(meta_path,"r", encoding="utf-8") as f:
                    meta = CASMeta(**json.load(f))
                expiry = meta.created_ts + meta.ttl_seconds
                if now >= expiry:
                    blob_path = meta_path[:-10]
                    for path in (blob_path, meta_path):
                        if os.path.exists(path):
                            os.remove(path); removed += 1
        return removed

CAS = ContentAddressableStore(root=os.getenv("IMU_CAS_ROOT","./.imu_cas"))

#=======old
@dataclass
class EvidenceMeta:
    source: str                    # URL/Path/Adapter
    retrieved_at: float            # epoch seconds
    ttl_seconds: int               # תוקף
    trust: float                   # 0..1
    content_type: str = "text/plain"
    signature: Optional[str] = None # מקום לחתימה, אם קיימת

def _hash_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def put_blob(content: bytes, meta: EvidenceMeta) -> str:
    h = _hash_bytes(content)
    path_blob = os.path.join(CAS_ROOT, h)
    path_meta = path_blob + ".json"
    if not os.path.exists(path_blob):
        with open(path_blob, "wb") as f: f.write(content)
    with open(path_meta, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)
    return h

def get_meta(hash_id: str) -> Optional[EvidenceMeta]:
    p = os.path.join(CAS_ROOT, hash_id + ".json")
    if not os.path.exists(p): return None
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    return EvidenceMeta(**d)

def get_blob(hash_id: str) -> Optional[bytes]:
    p = os.path.join(CAS_ROOT, hash_id)
    if not os.path.exists(p): return None
    with open(p, "rb") as f: return f.read()

def is_valid(hash_id: str, min_trust: float, now: Optional[float] = None) -> bool:
    now = now or time.time()
    m = get_meta(hash_id)
    if not m: return False
    if m.trust < min_trust: return False
    if now > m.retrieved_at + m.ttl_seconds: return False
    return True

def _ensure_root():
    os.makedirs(CAS_ROOT, exist_ok=True)

def put_bytes(b: bytes, meta: Dict[str, Any]) -> str:
    _ensure_root()
    h = hashlib.sha256(b).hexdigest()
    obj = os.path.join(CAS_ROOT, h)
    if not os.path.exists(obj):
        with open(obj, "wb") as f: f.write(b)
        with open(obj + ".meta.json", "w", encoding="utf-8") as f:
            meta = dict(meta or {})
            meta["ts"] = time.time()
            meta["sha256"] = h
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return h

def put_file(path: str, meta: Dict[str, Any]) -> str:
    with open(path, "rb") as f:
        cid = put_bytes(f.read(), {**meta, "path": path})
    return cid

def _sha256(data: bytes) -> str:
    h = hashlib.sha256(); h.update(data); return h.hexdigest()

class _CAS:
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