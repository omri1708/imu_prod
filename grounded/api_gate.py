# imu_repo/grounded/api_gate.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import urllib.request, json, time, ssl
from urllib.parse import urlencode
from grounded.source_policy import policy_singleton as SourcePolicy
from synth.schema_validate import validate_json_schema  # קיים משלב 36
from audit.cas import put_bytes
from audit.provenance_store import record_evidence

class ApiGateError(Exception): ...

class OfficialAPIGate:
    """
    אימות טענות מול API רשמי:
    - כופה allowlist (SourcePolicy)
    - HTTPS בלבד (אלא אם לוקאלי לבדיקה)
    - סכימת JSON
    - טריות לפי Last-Modified/Date מול TTL
    - חתימת גוף בתור ראיה ל-CAS + רישום ב-ledger
    """
    def __init__(self, *, ttl_s: float = 30*24*3600):
        self.ttl_s = float(ttl_s)

    def _check_url(self, url: str) -> None:
        if url.startswith("http://localhost") or url.startswith("http://127.0.0.1"):
            return
        if not url.startswith("https://"):
            raise ApiGateError("https_required")
        # Allowlist
        host = url.split("/")[2]
        if not SourcePolicy.allowed(host):
            raise ApiGateError(f"host_not_allowed:{host}")

    def _is_fresh(self, headers: Dict[str,Any]) -> bool:
        # TTL גס על בסיס Date / Last-Modified
        import email.utils as eut
        now = time.time()
        for k in ("Last-Modified","Date"):
            v = headers.get(k)
            if not v: 
                continue
            try:
                ts = time.mktime(eut.parsedate(v))
                return (now - ts) <= self.ttl_s
            except Exception:
                continue
        # אם אין מידע — נאפשר (ניתן להקשיח)
        return True

    def fetch(self, url: str, *, params: Dict[str,Any] | None=None, headers: Dict[str,str] | None=None, timeout: float=8.0) -> Tuple[bytes, Dict[str,str]]:
        self._check_url(url)
        if params:
            url = f"{url}?{urlencode(params)}"
        req = urllib.request.Request(url, headers=headers or {})
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            body = resp.read()
            hdrs = {k:v for k,v in resp.headers.items()}
        return body, hdrs

    def verify(self, *, name: str, url: str, json_schema: Dict[str,Any], claim_path: str, expected: Any,
               user_id: str, obj: str, tags: list[str] | None=None) -> Dict[str,Any]:
        body, hdrs = self.fetch(url)
        if not self._is_fresh(hdrs):
            raise ApiGateError("stale_content")
        # JSON parse
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception as e:
            raise ApiGateError(f"bad_json:{e}")
        # סכימה
        ok, errors = validate_json_schema(data, json_schema)
        if not ok:
            raise ApiGateError(f"schema_failed:{errors}")
        # בדיקת הטענה
        val = data
        for p in claim_path.split("."):
            val = val[p]
        if val != expected:
            raise ApiGateError(f"claim_mismatch:expected={expected} got={val}")
        # חתימת גוף ל-CAS + רישום ב-ledger
        cas = put_bytes(body, meta={"api": url, "name": name, "headers": hdrs})
        rec = record_evidence("official_api", {"url":url, "sha256": cas["sha256"], "headers": hdrs}, actor=f"user:{user_id}", obj=obj, tags=(tags or [])+["official_api"])
        return {"ok": True, "evidence": rec}