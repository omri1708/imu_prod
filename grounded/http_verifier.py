# grounded/http_verifier.py
from __future__ import annotations
import gzip
import http.client
import json, time, urllib.request, urllib.error, ssl
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from grounded.source_policy import policy_singleton as SourcePolicy
from sandbox.fs_net import NetSandbox, QuotaExceeded, SandboxViolation
from grounded.provenance import ProvenanceStore
from grounded.trust import classify_source, trust_score
from policy.policies import UserPolicy
from provenance.store import CAS

class ExternalVerifyError(Exception): ...
class HTTPVerifyError(Exception): ...

class HttpVerifier:
    def __init__(self, cas: CAS):
        self.cas = cas
        self._ssl_ctx = ssl.create_default_context()

    def fetch_json(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": "IMU/grounded"})
        with urllib.request.urlopen(req, timeout=timeout, context=self._ssl_ctx) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} {url}")
            data = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "json" not in ctype:
                raise RuntimeError(f"Non-JSON response: {ctype}")
            return json.loads(data.decode("utf-8"))

    def verify_claim(self, user_policy: UserPolicy, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        claim = {"subject":"...", "predicate":"...", "object":"...", "source":{"kind":"web","url":"..."}, "domain":"finance"}
        Returns a dict with {'ok':bool, 'reason':..., 'evidence_digest':..., 'trust':..., 'staleness_ok':...}
        """
        src = claim.get("source", {})
        url = src.get("url")
        if not url:
            return {"ok": False, "reason": "missing_source_url"}
        data = self.fetch_json(url)
        # NOTE: domain-specific validation would go here (schemas, unit ranges etc.)
        content = json.dumps({"url": url, "data": data}, sort_keys=True).encode("utf-8")
        digest = self.cas.put(content, kind="evidence", trust=user_policy.min_trust.get(src.get("kind","external"), "external"),
                              extra_meta={"url": url, "domain": claim.get("domain","generic")})
        # TTL/staleness check:
        max_stale = user_policy.max_staleness.get(claim.get("domain","docs"), 365*24*3600)
        now = time.time()
        staleness_ok = True
        # If data has timestamp, prefer it
        ts = data.get("timestamp") if isinstance(data, dict) else None
        if ts:
            try:
                ts = float(ts)
                staleness_ok = (now - ts) <= max_stale
            except Exception:
                staleness_ok = True
        return {"ok": staleness_ok, "reason": "verified" if staleness_ok else "stale",
                "evidence_digest": digest, "trust": user_policy.min_trust.get(src.get("kind","external"), "external"),
                "staleness_ok": staleness_ok}


def http_get_json(url: str, timeout_s: float = 4.0) -> Dict[str,Any]:
    if not SourcePolicy.domain_allowed(url):
        raise ExternalVerifyError("domain_not_allowed")
    u = urlparse(url)
    host = u.hostname
    port = u.port or (443 if u.scheme=="https" else 80)
    path = u.path or "/"
    if u.query:
        path += "?" + u.query
    if u.scheme == "https":
        conn = http.client.HTTPSConnection(host, port, timeout=timeout_s)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    conn.request("GET", path, headers={"Accept":"application/json"})
    r = conn.getresponse()
    data = r.read()
    try:
        obj = json.loads(data.decode("utf-8","replace"))
    except Exception:
        obj = {"raw": data.decode("utf-8","replace")}
    return {"status": r.status, "headers": dict(r.getheaders()), "body": obj, "ts": time.time()}


def fetch_and_record(key: str, url: str, pv: ProvenanceStore, ns: NetSandbox) -> Dict[str,Any]:
    try:
        resp = ns.http_get(url, timeout_s=5.0)
        status = int(resp["status"])
        body = resp["body"]
        # מנסה לפענח JSON; אם לא — שומר raw (מכווץ)
        parsed: Any
        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception:
            parsed = {"raw_gzip": gzip.compress(body).hex(), "len": len(body)}
        rec = {"status": status, "url": url, "payload": parsed}
        # קביעת אמון לפי class
        cls = classify_source(url); base_trust = trust_score(cls)
        return pv.put(key, rec, source_url=url, trust=base_trust)
    except (QuotaExceeded, SandboxViolation) as e:
        raise HTTPVerifyError(f"sandbox:{e}")
    except Exception as e:
        raise HTTPVerifyError(f"http_error:{e}")