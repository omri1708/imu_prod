# imu_repo/grounded/source_policy.py
from __future__ import annotations
from typing import Dict, List, Optional
import time, hmac, hashlib

class SourcePolicy:
    def __init__(self):
        self.allow_domains: List[str] = ["example.com", "example.org", "internal.test"]
        self.default_ttl_s: int = 24*3600
        self.domain_ttl: Dict[str,int] = {}
        self.hmac_key: bytes = b"imu_default_key_change_me"

    def domain_allowed(self, url_or_domain: str) -> bool:
        d = url_or_domain
        if "://" in d:
            d = d.split("://",1)[1].split("/",1)[0]
        d = d.lower()
        return any(d.endswith(ad) for ad in self.allow_domains)

    def set_allowlist(self, domains: List[str]): self.allow_domains = [d.lower() for d in domains]
    def set_ttl(self, default_ttl_s: int, domain_ttl: Optional[Dict[str,int]] = None):
        self.default_ttl_s = int(default_ttl_s)
        if domain_ttl:
            self.domain_ttl = {k.lower(): int(v) for k,v in domain_ttl.items()}
    def set_trust_threshold(self, thr: float) -> None:
        self.trust_threshold = float(thr)
    def allowed_domains(self) -> List[str]:
        return list(self.allow_domains)

    def ttl_for(self, url_or_domain: str) -> int:
        d = url_or_domain
        if "://" in d:
            d = d.split("://",1)[1].split("/",1)[0]
        d = d.lower()
        for dom, ttl in self.domain_ttl.items():
            if d.endswith(dom):
                return ttl
        return self.default_ttl_s

    def sign_blob(self, payload: bytes) -> str:
        return hmac.new(self.hmac_key, payload, hashlib.sha256).hexdigest()
    def verify_blob(self, payload: bytes, hexdigest: str) -> bool:
        return hmac.compare_digest(self.sign_blob(payload), hexdigest)

policy_singleton = SourcePolicy()