# grounded/http_verifier.py
from __future__ import annotations
import gzip
import ssl, http.client, json, time
from typing import Dict, Any
from urllib.parse import urlparse

from grounded.source_policy import policy_singleton as SourcePolicy
from sandbox.fs_net import NetSandbox, QuotaExceeded, SandboxViolation
from grounded.provenance import ProvenanceStore
from grounded.trust import classify_source, trust_score


class ExternalVerifyError(Exception): ...
class HTTPVerifyError(Exception): ...

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