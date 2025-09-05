# provenance/store.py
# -*- coding: utf-8 -*-
import os, time, json, hashlib, base64
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from adapters.contracts import ResourceRequired
import hashlib, json, os, shutil, time
from pathlib import Path
from typing import Optional, Dict
from contracts.base import Artifact
from typing import Dict, Any, Optional, Tuple
import hashlib, json, os, time, threading


class ResourceRequired(RuntimeError):
    def __init__(self, what:str, how:str):
        super().__init__(f"resource_required: {what}\nhow_to_provide: {how}")
        self.what=what; self.how=how

CAS_ROOT = os.environ.get("IMU_CAS_ROOT","./_imu_cas")
os.makedirs(CAS_ROOT, exist_ok=True)

@dataclass
class Evidence:
    """רישום ראיה: hash, מקור, חותמת-זמן, רמת אמון [0..1], חתימה לוגית"""
    algo: str         # 'sha256'
    digest: str       # hex
    source: str       # URL/file/“calc”
    ts: float         # epoch
    trust: float      # [0..1]
    signature: str    # חתימת מקור/הפקה (לוגית: sha256(source+digest+ts))

def sha256_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()

def sha256_file(path:str)->str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()

def _sign(source:str, digest:str, ts:float)->str:
    return hashlib.sha256((source+digest+str(ts)).encode()).hexdigest()

def cas_put_bytes(b:bytes)->str:
    d = sha256_bytes(b)
    p = os.path.join(CAS_ROOT, d)
    if not os.path.exists(p):
        with open(p,'wb') as f: f.write(b)
    return d

def cas_put_file(path:str)->str:
    d = sha256_file(path)
    dst = os.path.join(CAS_ROOT, d)
    if not os.path.exists(dst):
        with open(path,'rb') as src, open(dst,'wb') as out:
            out.write(src.read())
    return d

def evidence_from_bytes(b:bytes, source:str, trust:float)->Evidence:
    ts=time.time(); d=sha256_bytes(b)
    return Evidence("sha256", d, source, ts, trust, _sign(source,d,ts))

def evidence_from_file(path:str, source:str, trust:float)->Evidence:
    ts=time.time(); d=sha256_file(path)
    return Evidence("sha256", d, source, ts, trust, _sign(source,d,ts))

def write_ledger(record:Dict, ledger_path:str="./_imu_cas/ledger.jsonl"):
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    with open(ledger_path,'a',encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False)+"\n")
class ProvenanceStore:
    """
    Content-Addressable Store (CAS) + רמות אמון + TTL.
    כל פריט נשמר לפי sha256(content), עם מטא: source, trust, ttl, created_at, signatures.
    """
    def __init__(self, root=".imu_data/prov", default_ttl_s=30*24*3600):
        self.root=root; os.makedirs(self.root, exist_ok=True)
        self.default_ttl_s=default_ttl_s
        self._lock=threading.RLock()

    def _path(self, digest:str)->str:
        return os.path.join(self.root, digest[:2], digest[2:])
    def _meta_path(self, digest:str)->str:
        return self._path(digest)+".meta.json"

    def put(self, content:bytes, source:str, trust:int, ttl_s:Optional[int]=None,
            evidence:Optional[Dict[str,Any]]=None, signatures:Optional[Dict[str,str]]=None) -> str:
        """
        trust ∈ {0..100}; TTL קשיח; evidence: מילון ראיות (כגון URL/sha/headers); signatures: חתימות מקור (אם קיימות).
        """
        digest=hashlib.sha256(content).hexdigest()
        p=self._path(digest); mp=self._meta_path(digest)
        with self._lock:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if not os.path.exists(p):
                with open(p,"wb") as f: f.write(content)
            meta={
              "digest":digest,
              "source":source,
              "trust":int(trust),
              "ttl_s": int(self.default_ttl_s if ttl_s is None else ttl_s),
              "created_at": int(time.time()),
              "evidence": evidence or {},
              "signatures": signatures or {}
            }
            with open(mp,"w",encoding="utf-8") as f: json.dump(meta,f,ensure_ascii=False,indent=2)
        return digest

    def get(self, digest:str, min_trust:int=0) -> Tuple[bytes, Dict[str,Any]]:
        p=self._path(digest); mp=self._meta_path(digest)
        if not (os.path.exists(p) and os.path.exists(mp)):
            raise FileNotFoundError(digest)
        with self._lock:
            with open(mp,"r",encoding="utf-8") as f: meta=json.load(f)
            now=int(time.time())
            if now - int(meta["created_at"]) > int(meta["ttl_s"]):
                # פג־תוקף – מוחקים קשיח
                try: os.remove(p)
                except: pass
                try: os.remove(mp)
                except: pass
                raise RuntimeError(f"expired:{digest}")
            if meta["trust"] < min_trust:
                raise RuntimeError(f"insufficient_trust:{meta['trust']}< {min_trust}")
            with open(p,"rb") as f: content=f.read()
        return content, meta

    def verify_chain(self, digest:str, min_trust:int=50, require_signature:bool=False)->bool:
        _, meta=self.get(digest, min_trust=min_trust)
        if require_signature and not meta.get("signatures"):
            return False
        return True

def _ensure_keys(key_dir: str):
    try:
        from nacl.signing import SigningKey
    except Exception:
        raise ResourceRequired("PyNaCl (ed25519)", "pip install pynacl")
    os.makedirs(key_dir, exist_ok=True)
    skf = os.path.join(key_dir, "ed25519.sk")
    pkf = os.path.join(key_dir, "ed25519.pk")
    if not os.path.exists(skf):
        sk = SigningKey.generate()
        with open(skf, "wb") as f: f.write(sk.encode())
        with open(pkf, "wb") as f: f.write(sk.verify_key.encode())
    else:
        with open(skf, "rb") as f: sk = SigningKey(f.read())
    return sk

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@dataclass
class EvidenceMeta:
    url: Optional[str]
    fetched_ts: float
    sha256: str
    trust: float  # 0..1
    not_before_ts: Optional[float] = None
    not_after_ts: Optional[float] = None
    sig_b64: Optional[str] = None

class CASStore:
    def __init__(self, root_dir: str = ".imu/cas", key_dir: str = ".imu/keys"):
        self.root_dir = root_dir
        self.key_dir = key_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _path(self, digest: str) -> str:
        return os.path.join(self.root_dir, digest)

    def put_bytes(self, b: bytes, sign: bool = True, url: str = None, trust: float = 0.5,
                  not_after_days: int = 365) -> EvidenceMeta:
        h = _hash_bytes(b)
        p = self._path(h)
        if not os.path.exists(p):
            with open(p, "wb") as f: f.write(b)
        meta = EvidenceMeta(url=url, fetched_ts=time.time(), sha256=h, trust=float(trust))
        if not_after_days:
            meta.not_after_ts = meta.fetched_ts + not_after_days*24*3600
        if sign:
            try:
                from nacl.signing import SigningKey
            except Exception:
                raise ResourceRequired("PyNaCl (ed25519)", "pip install pynacl")
            sk = _ensure_keys(self.key_dir)
            sig = sk.sign(h.encode("utf-8")).signature
            meta.sig_b64 = base64.b64encode(sig).decode("ascii")
        # שמירת מטא-דאטה
        with open(p + ".json", "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, ensure_ascii=False, indent=2)
        return meta

    def get(self, digest: str) -> Optional[EvidenceMeta]:
        p = self._path(digest) + ".json"
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return EvidenceMeta(**d)

    def verify_meta(self, meta: EvidenceMeta) -> bool:
        # תוקף זמן + חתימה
        now = time.time()
        if meta.not_before_ts and now < meta.not_before_ts:
            return False
        if meta.not_after_ts and now > meta.not_after_ts:
            return False
        if meta.sig_b64:
            try:
                from nacl.signing import VerifyKey
            except Exception:
                raise ResourceRequired("PyNaCl (ed25519)", "pip install pynacl")
            pkf = os.path.join(self.key_dir, "ed25519.pk")
            if not os.path.exists(pkf):
                return False
            with open(pkf, "rb") as f: vk = VerifyKey(f.read())
            try:
                vk.verify(meta.sha256.encode("utf-8"), base64.b64decode(meta.sig_b64))
            except Exception:
                return False
        return True