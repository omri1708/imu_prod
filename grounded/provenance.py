# imu_repo/grounded/provenance.py
from __future__ import annotations
import os, json, time, hmac, hashlib
from typing import Dict, Any, Optional, List, Tuple
from grounded.source_policy import policy_singleton as Policy
from grounded.trust import classify_source, trust_score
import re
from grounded.evidence_store import EvidenceStore


def _pick_root() -> str:
    """
    בוחר שורש כתיב ללא כתיבה בזמן import:
    1) IMU_ROOT אם הוגדר והוא כתיב
    2) ./assurance_store_text
    3) תיקיית הפרויקט (dir של הקובץ הזה ../)
    4) cwd
    """
    env = os.environ.get("IMU_ROOT")
    candidates = []
    if env:
        candidates.append(os.path.abspath(env))
    candidates.append(os.path.abspath("./assurance_store_text"))
    candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    candidates.append(os.getcwd())
    for c in candidates:
        try:
            os.makedirs(c, exist_ok=True)
            return c
        except Exception:
            continue
    return os.getcwd()

ROOT = _pick_root()
STORE = os.path.join(ROOT, ".provenance")
LOGS = os.path.join(ROOT, "logs")
PROV_LOG = os.path.join(LOGS, "provenance.jsonl")

def _ensure_dirs():
    # יצירה עצלה בזמן ריצה, לא בזמן import
    os.makedirs(STORE, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)


def _canonical(d: Dict[str,Any]) -> bytes:
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",",":")).encode("utf-8")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sign_evidence(secret: bytes, payload: Dict[str,Any]) -> Dict[str,Any]:
    """
    מחזיר רשומה עם sha256 + hmac חתום וחותמת זמן. לא משנה את payload המקורי.
    """
    ts = int(time.time())
    body = dict(payload)
    body.setdefault("ts", ts)
    blob = _canonical(body)
    digest = _sha256_bytes(blob)
    sig = hmac.new(secret, digest.encode("utf-8"), hashlib.sha256).hexdigest()
    record = {
        "sha256": digest,
        "sig_hmac_sha256": sig,
        "payload": body
    }
    return record


def persist_record(record: Dict[str,Any]) -> str:
    """
    שומר JSON עם שם הקובץ לפי sha256 (CAS) + רושם שורת audit.
    מחזיר נתיב הקובץ.
    """
    _ensure_dirs
    sha = record["sha256"]
    path = os.path.join(STORE, f"{sha}.json")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False))
    with open(PROV_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": int(time.time()),
            "sha256": sha,
            "sig": record["sig_hmac_sha256"]
        }, ensure_ascii=False) + "\n")
    return path


def verify_signature(secret: bytes, record: Dict[str,Any]) -> bool:
    sha = record.get("sha256")
    sig = record.get("sig_hmac_sha256")
    if not sha or not sig: return False
    exp = hmac.new(secret, sha.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(exp, sig)


def _sha(b: bytes)->str: return hashlib.sha256(b).hexdigest()


def _validate_simple_schema(claim: Dict[str,Any]) -> Tuple[bool,str]:
    """
    ולידציה אופציונלית לטענה:
    claim["schema"] יכול להיות:
      {"type":"number","value": <float>, "min":0, "max":100}
      {"type":"string","value": "<str>", "pattern": "^[A-Z].+"}
    """
    sch = claim.get("schema")
    if not sch: return (True, "")
    t = sch.get("type")
    if t=="number":
        try:
            v = float(sch["value"])
        except Exception:
            return (False, "schema_number_invalid")
        if "min" in sch and v < float(sch["min"]): return (False, "schema_min_violation")
        if "max" in sch and v > float(sch["max"]): return (False, "schema_max_violation")
        return (True, "")
    if t=="string":
        v = str(sch.get("value",""))
        pat = sch.get("pattern")
        if pat and not re.match(pat, v):
            return (False, "schema_pattern_violation")
        return (True, "")
    return (False, "schema_unknown_type")


def validate_claim(claim: Dict[str,Any],
                   store: EvidenceStore,
                   *, ttl_s: float | None=None,
                   allowed_domains: Optional[List[str]]=None,
                   require_sig: bool=True,
                   now: Optional[float]=None) -> Dict[str,Any]:
    """
    בודק:
      - יש לפחות עדות אחת
      - כל עדות עוברת verify (דומיין/חתימה/תפוגה)
      - schema אופציונלי תקין
      - לפחות עדות *אחת* טובה לכל claim (ברירת מחדל), אפשר לדרוש כולן דרך require_all=True
    """
    evid = claim.get("evidence") or []
    if not evid:
        return {"ok": False, "reason": "no_evidence", "claim": claim}
    evid_res=[]
    ok_any=False
    for sha in evid:
        r = store.verify(sha, allowed_domains=allowed_domains, require_sig=require_sig, now=now)
        evid_res.append(r)
        ok_any = ok_any or r.get("ok", False)
    sch_ok, sch_reason = _validate_simple_schema(claim)
    ok = ok_any and sch_ok
    return {"ok": ok, "claim": claim, "evidence_results": evid_res,
            "schema_ok": sch_ok, "schema_reason": sch_reason}


class ProvenanceStore:
    """
    Content-addressable evidence with minimal signing (HMAC over bytes).
    """
    def __init__(self, root_dir: str | None = None):
        self.root = os.path.abspath(root_dir or STORE)
        os.makedirs(self.root, exist_ok=True)
        self.key_path = os.path.join(root_dir, ".hmac_key")
        if not os.path.exists(self.key_path):
            # מפתח מקומי — עבור חתימה פנימית בלבד
            k = os.urandom(32)
            with open(self.key_path,"wb") as f: f.write(k)

    def _key(self)->bytes:
        with open(self.key_path,"rb") as f: return f.read()

    def put(self, key: str, obj: Dict[str,Any], source_url: str, trust: float = 0.5) -> Dict[str,Any]:
        label = classify_source(source_url)
        rec = {
            "key": key, "source_url": source_url, "trust": float(trust),
            "class": label, "class_score": trust_score(label),
            "ts": time.time(), "payload": obj
        }
        b = json.dumps(rec, ensure_ascii=False, sort_keys=True).encode("utf-8")
        rec["sig_hmac_sha256"] = Policy.sign_blob(b)
        path = os.path.join(self.root, f"{key}.prov.json")
        with open(path,"w",encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return rec

    def get(self, key: str) -> Optional[Dict[str,Any]]:
        path = os.path.join(self.root, f"{key}.prov.json")
        if not os.path.exists(path): return None
        with open(path,"r",encoding="utf-8") as f:
            rec = json.load(f)
        tmp = dict(rec); sig = tmp.pop("sig_hmac_sha256", "")
        b = json.dumps(tmp, ensure_ascii=False, sort_keys=True).encode("utf-8")
        rec["_sig_ok"] = Policy.verify_blob(b, sig)
        ttl = Policy.ttl_for(rec.get("source_url","internal.test"))
        rec["_fresh"] = (time.time() - float(rec.get("ts",0))) <= ttl
        return rec

    def verify(self, hash_: str) -> bool:
        rec = self.get(hash_) or {}
        obj = rec.get("obj")
        meta = rec.get("meta") or {}
        if obj is None or "sig" not in meta: return False
        b = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()
        sig = hmac.new(self._key(), b, hashlib.sha256).hexdigest()
        return sig == meta.get("sig")