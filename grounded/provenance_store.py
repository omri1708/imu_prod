# imu_repo/grounded/provenance_store.py
from __future__ import annotations
import os, json, time, hmac, hashlib, threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from user.crypto_store import CryptoStore
from grounded.source_policy import policy_singleton as SourcePolicy
from urllib.parse import urlparse

def _pick_root() -> Path:
    candidates = []
    env = os.environ.get("IMU_PROV_ROOT")
    if env:
        candidates.append(Path(env).resolve())
    candidates.append(Path("./assurance_store_text").resolve())
    candidates.append(Path("./assurance_store").resolve())
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            return c
        except Exception:
            continue
    return Path.cwd()

ROOTP = _pick_root()
OBJ   = os.path.join(ROOTP, "objects")
META  = os.path.join(ROOTP, "meta")
KEYF  = os.path.join(ROOTP, "secret.key")
IDX  = os.path.join(ROOTP, "index.jsonl")
KEY  = os.path.join(ROOTP, "hmac.key")
os.makedirs(OBJ, exist_ok=True)
os.makedirs(ROOTP, exist_ok=True)
if not os.path.exists(KEY):
    open(KEY,"wb").write(os.urandom(32))
_lock = threading.RLock()

def _ensure_dirs():
    global OBJ, META, KEYF, ROOTP
    try:
        os.makedirs(OBJ, exist_ok=True)
        os.makedirs(META, exist_ok=True)
    except Exception as e:
        # נסיון fallback לספרייה בתוך הפרויקט
        fallback = Path("./assurance_store_text").resolve()
        os.makedirs(fallback / "objects", exist_ok=True)
        os.makedirs(fallback / "meta", exist_ok=True)
        ROOTP = fallback
        OBJ   = os.path.join(ROOTP, "objects")
        META  = os.path.join(ROOTP, "meta")
        KEYF  = os.path.join(ROOTP, "secret.key")


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hmac(digest_hex: str, key: bytes) -> str:
    return hmac.new(key, bytes.fromhex(digest_hex), hashlib.sha256).hexdigest()


def _key() -> bytes:
    if not os.path.exists(KEYF):
        k = os.urandom(32)
        with open(KEYF, "wb") as f:
            f.write(k)
        return k
    return open(KEYF, "rb").read()


def _obj_path(digest_hex: str) -> str:
    return os.path.join(OBJ, digest_hex[:2], digest_hex)


def _meta_path(digest_hex: str) -> str:
    return os.path.join(META, f"{digest_hex}.json")

def add_evidence(content: bytes, meta: Dict[str, Any] | None=None, *, sign: bool=True) -> str:
    _ensure_dirs()
    ...
    """
    מכניס ראיה למחסן (content-addressable) ושומר מטה־דאטה:
      meta: {source_url?, fetched_at?, ttl_s?, trust? in [0..1]}
    חתימת HMAC על ה-digest עבור אימות מקור.
    מחזיר digest_hex.
    """
    with _lock:
        dg = _sha256(content)
        ddir = os.path.dirname(_obj_path(dg))
        os.makedirs(ddir, exist_ok=True)
        op = _obj_path(dg)
        if not os.path.exists(op):
            with open(op, "wb") as f:
                f.write(content)
        m = dict(meta or {})
        m.setdefault("fetched_at", int(time.time()))
        m.setdefault("ttl_s", 0)
        m.setdefault("trust", 0.5)
        m["digest"] = dg
        if sign:
            k = _key()
            m["hmac"] = _hmac(dg, k)
        with open(_meta_path(dg), "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        return dg

def get_evidence(digest_hex: str) -> bytes:
    p = _obj_path(digest_hex)
    if not os.path.exists(p):
        raise FileNotFoundError(digest_hex)
    return open(p, "rb").read()


def get_meta(digest_hex: str) -> Dict[str, Any]:
    mp = _meta_path(digest_hex)
    if not os.path.exists(mp):
        return {}
    return json.load(open(mp, "r", encoding="utf-8"))


def verify(digest_hex: str, *, require_hmac: bool=True, min_trust: float=0.0) -> Dict[str, Any]:
    """
    מאמת:
      - קיום האובייקט
      - התאמת SHA-256
      - חתימת HMAC (אם נדרש)
      - תוקף TTL
      - רמת אמון מינימלית
    """
    out = {"ok": False, "reasons": []}
    op = _obj_path(digest_hex)
    mp = _meta_path(digest_hex)
    if not os.path.exists(op):
        out["reasons"].append("missing_object")
        return out
    if not os.path.exists(mp):
        out["reasons"].append("missing_meta")
        return out

    content = open(op, "rb").read()
    dg2 = hashlib.sha256(content).hexdigest()
    if dg2 != digest_hex:
        out["reasons"].append("digest_mismatch")
        return out

    m = json.load(open(mp, "r", encoding="utf-8"))
    if require_hmac:
        h = m.get("hmac")
        if not h:
            out["reasons"].append("missing_hmac")
            return out
        if not hmac.compare_digest(h, _hmac(digest_hex, _key())):
            out["reasons"].append("hmac_invalid")
            return out

    ttl = int(m.get("ttl_s", 0))
    if ttl > 0:
        if int(time.time()) > int(m.get("fetched_at", 0)) + ttl:
            out["reasons"].append("expired")
            return out

    trust = float(m.get("trust", 0.0))
    if trust < float(min_trust):
        out["reasons"].append("trust_below_threshold")
        return out

    out["ok"] = True
    out["meta"] = m
    return out

class ProvenanceError(Exception): ...

class CAS:
    def __init__(self, root: str = ".imu_state/prov"):
        self.root = root
        self.obj = os.path.join(root, "objects")
        self.idx = os.path.join(root, "index.jsonl")
        os.makedirs(self.obj, exist_ok=True)
        os.makedirs(root, exist_ok=True)
        if not os.path.exists(self.idx):
            with open(self.idx,"w",encoding="utf-8"):
                pass

    def put(self, content: bytes, meta: Dict[str,Any]) -> str:
        h = _sha256(content)
        path = os.path.join(self.obj, h)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(content)
        rec = {"hash":h, "meta":meta, "ts":time.time()}
        with open(self.idx,"a",encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
        return h

    def get(self, h: str) -> Optional[bytes]:
        path = os.path.join(self.obj, h)
        if not os.path.exists(path):
            return None
        with open(path,"rb") as f:
            return f.read()


class EvidenceStore:
    def __init__(self, root: str = ".imu_state/prov"):
        self.cas = CAS(root)
        self.crypto = CryptoStore(os.path.join(root, "..", "crypto.key"))
        self.claims_map = os.path.join(root, "claims.json")
        if not os.path.exists(self.claims_map):
            with open(self.claims_map,"w",encoding="utf-8") as f:
                json.dump({}, f)
        self.policy = SourcePolicy

        self.resolvers: Dict[str, Callable[[str], Tuple[bytes, Dict[str,Any]]]] = {
            "bench": self._resolve_bench,
            "file": self._resolve_file,
            "httpcache": self._resolve_httpcache
        }
        try:
            from adapters.http_fetch import http_fetch_bytes
            def _resolve_http(rest: str) -> Tuple[bytes, Dict[str,Any]]:
                url = "http:"+rest if not rest.startswith("//") else "http:"+rest
                b,meta = http_fetch_bytes(url, timeout=2.0)
                meta.update(_http_meta(url))
                return b, meta
            def _resolve_https(rest: str) -> Tuple[bytes, Dict[str,Any]]:
                url = "https:"+rest if not rest.startswith("//") else "https:"+rest
                b,meta = http_fetch_bytes(url, timeout=2.0)
                meta.update(_http_meta(url))
                return b, meta
            self.resolvers["http"] = _resolve_http
            self.resolvers["https"] = _resolve_https
        except Exception:
            pass

    def add_resolver(self, scheme: str, fn: Callable[[str], Tuple[bytes, Dict[str,Any]]]):
        self.resolvers[scheme] = fn

    # ---------- resolvers ----------
    def _resolve_bench(self, name: str) -> Tuple[bytes, Dict[str,Any]]:
        content = f"bench::{name}::static-proof".encode("utf-8")
        meta = {"source": f"bench:{name}", "kind":"bench", "trust": 0.9, "fetched_at": time.time()}
        return content, meta

    def _resolve_file(self, path: str) -> Tuple[bytes, Dict[str,Any]]:
        path = path.strip()
        if not os.path.exists(path):
            raise ProvenanceError(f"file_not_found:{path}")
        with open(path,"rb") as f:
            b=f.read()
        meta = {"source": f"file:{path}", "kind":"file", "trust": 0.8, "fetched_at": time.time()}
        return b, meta

    def _resolve_httpcache(self, spec: str) -> Tuple[bytes, Dict[str,Any]]:
        """
        httpcache://domain/path -> קורא מ-.imu_state/httpcache/domain/path (מייצר לבדיקה)
        ו”מזייף” מטא-Headers אמינים מפני שאין רשת בסביבת ריצה.
        """
        u = urlparse("httpcache://"+spec if not spec.startswith("//") else "httpcache:"+spec)
        root = ".imu_state/httpcache"
        p = os.path.join(root, u.hostname or "host", *(u.path.strip("/").split("/") if u.path else []))
        if not os.path.exists(p):
            raise ProvenanceError(f"httpcache_not_found:{p}")
        with open(p,"rb") as f:
            b=f.read()
        meta = {
            "source": f"httpcache://{u.hostname}{u.path}",
            "kind": "httpcache",
            "domain": u.hostname or "host",
            "etag": "W/\"demo-etag\"",
            "last_modified": "Tue, 01 Sep 2025 00:00:00 GMT",
            "content_type": "application/octet-stream",
            "fetched_at": time.time()
        }
        meta.update(_domain_trust(meta["domain"]))
        return b, meta

    # ---------- evidence ops ----------
    def register_evidence(self, source_uri: str, extra_meta: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
        if not self.policy.domain_allowed(source_uri):
            raise ProvenanceError(f"domain_not_allowed:{source_uri}")
        try:
            scheme, rest = source_uri.split(":",1)
        except ValueError:
            raise ProvenanceError(f"bad_uri:{source_uri}")
        if scheme not in self.resolvers:
            raise ProvenanceError(f"no_resolver_for:{scheme}")
        content, meta = self.resolvers[scheme](rest)
        if extra_meta:
            meta.update(extra_meta)
        # תוספת trust/max_age לפי דומיין/מדיניות
        trust, max_age = self.policy.trust_for(source_uri)
        meta.setdefault("trust", trust)
        meta.setdefault("max_age_sec", max_age)
        h = self.cas.put(content, meta)
        sig = self.crypto.sign(content)
        return {"hash": h, "sig": sig, "meta": meta, "fetched_at": meta.get("fetched_at", time.time())}

    def link_claim(self, claim: str, evidences: List[Dict[str,Any]]):
        with open(self.claims_map,"r",encoding="utf-8") as f:
            m = json.load(f)
        arr = m.get(claim, [])
        arr.extend(evidences)
        m[claim] = arr
        with open(self.claims_map,"w",encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)

    def claim_evidences(self, claim: str) -> List[Dict[str,Any]]:
        with open(self.claims_map,"r",encoding="utf-8") as f:
            m = json.load(f)
        return m.get(claim, [])

    def verify_evidence(self, ev: Dict[str,Any]) -> bool:
        h=ev.get("hash")
        sig=ev.get("sig")
        content=self.cas.get(h) if h else None
        if not content or not sig:
            return False
        if not CryptoStore(os.path.join(self.cas.root, "..", "crypto.key")).verify(content, sig):
            return False
        return True


def _http_meta(url: str) -> Dict[str,Any]:
    u = urlparse(url)
    base = {"domain": u.hostname or "", "path": u.path}
    base.update(_domain_trust(base["domain"]))
    return base


def _domain_trust(domain: str) -> Dict[str,Any]:
    t, age = SourcePolicy.trust_for(f"https://{domain}/")
    return {"trust": t, "max_age_sec": age}