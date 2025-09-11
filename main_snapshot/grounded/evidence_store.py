# imu_repo/grounded/evidence_store.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import os, time, json, hashlib, hmac, urllib.parse

ROOT = "/mnt/data/imu_repo/evidence"
BLOBS = os.path.join(ROOT, "blobs")
META  = os.path.join(ROOT, "meta")
LOG   = os.path.join(ROOT, "audit.log")
SECRET_FILE = os.path.join(ROOT, "secret.key")

def _now() -> float:
    return time.time()

def _ensure_dirs():
    os.makedirs(BLOBS, exist_ok=True)
    os.makedirs(META, exist_ok=True)
    os.makedirs(ROOT, exist_ok=True)
    if not os.path.exists(SECRET_FILE):
        with open(SECRET_FILE,"wb") as f:
            f.write(os.urandom(32))

def _read_secret() -> bytes:
    with open(SECRET_FILE,"rb") as f:
        return f.read()

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _sign(b: bytes, key: bytes) -> str:
    return hmac.new(key, b, hashlib.sha256).hexdigest()

def _domain_from_url(url: str) -> str:
    try:
        u = urllib.parse.urlparse(url)
        return (u.hostname or "").lower()
    except Exception:
        return ""

def _log(ev: Dict[str,Any]) -> None:
    ev = dict(ev); ev["ts"] = _now()
    with open(LOG,"a",encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")

class EvidenceStore:
    """
    Content-addressable evidence:
      - blob נשמר לפי sha256 בתיקיית blobs/
      - meta JSON ב-meta/{sha}.json כולל: url, domain, type, size, stored_at, ttl_s, expires_at, sig
      - חתימה HMAC עם מפתח מקומי (secret.key)
      - verify(): בדיקת sha/חתימה/תפוגה/דומיין
    """
    def __init__(self, root: str=ROOT):
        self.root = root
        _ensure_dirs()
        self.secret = _read_secret()

    def put(self, *, source_url: str, content: bytes | str,
            content_type: str="text/plain", ttl_s: float=7*24*3600,
            stored_at: Optional[float]=None) -> str:
        if isinstance(content, str):
            content = content.encode("utf-8")
        sha = _sha256(content)
        blob_p = os.path.join(BLOBS, sha)
        if not os.path.exists(blob_p):
            with open(blob_p,"wb") as f:
                f.write(content)
        meta = {
            "sha256": sha,
            "source_url": source_url,
            "domain": _domain_from_url(source_url),
            "content_type": content_type,
            "size": len(content),
            "stored_at": float(stored_at) if stored_at is not None else _now(),
            "ttl_s": float(ttl_s)
        }
        meta["expires_at"] = meta["stored_at"] + meta["ttl_s"]
        meta["sig"] = _sign(content, self.secret)
        with open(os.path.join(META, f"{sha}.json"),"w",encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        _log({"op":"put","sha":sha,"url":source_url,"size":len(content)})
        return sha

    def get_meta(self, sha: str) -> Dict[str,Any] | None:
        p = os.path.join(META, f"{sha}.json")
        if not os.path.exists(p): return None
        return json.load(open(p,"r",encoding="utf-8"))

    def open_blob(self, sha: str) -> bytes:
        with open(os.path.join(BLOBS, sha),"rb") as f:
            return f.read()

    def verify(self, sha: str, *,
               allowed_domains: Optional[list[str]]=None,
               require_sig: bool=True,
               now: Optional[float]=None) -> Dict[str,Any]:
        meta = self.get_meta(sha)
        if not meta:
            return {"ok": False, "reason": "meta_missing"}
        blob_p = os.path.join(BLOBS, sha)
        if not os.path.exists(blob_p):
            return {"ok": False, "reason": "blob_missing"}

        b = self.open_blob(sha)
        sha_ok = (_sha256(b) == sha)
        sig_ok = True
        if require_sig:
            sig_ok = (_sign(b, self.secret) == meta.get("sig"))
        tnow = _now() if now is None else float(now)
        fresh = tnow <= float(meta.get("expires_at", 0))
        dom_ok = True
        if allowed_domains:
            dom_ok = (meta.get("domain","") in [d.lower() for d in allowed_domains])

        ok = sha_ok and sig_ok and fresh and dom_ok
        res = {"ok": ok, "sha": sha, "sha_ok": sha_ok, "sig_ok": sig_ok,
               "fresh": fresh, "domain_ok": dom_ok, "meta": meta}
        _log({"op":"verify","sha":sha,"ok":ok,"sha_ok":sha_ok,"sig_ok":sig_ok,"fresh":fresh,"domain_ok":dom_ok})
        return res