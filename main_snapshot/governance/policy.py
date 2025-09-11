# governance/policy.py
# -*- coding: utf-8 -*-
import time, urllib.parse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from contracts.errors import ContractViolation, PolicyDenied
from contracts.schema import validate_schema

@dataclass
class EvidenceRule:
    min_trust: float = 0.7
    max_age_sec: int = 7*24*3600
    allowed_domains: List[str] = field(default_factory=lambda: [])
    require_signature: bool = True

@dataclass
class RespondPolicy:
    require_claims: bool = True
    require_evidence: bool = True
    evidence: EvidenceRule = field(default_factory=EvidenceRule)
    allow_math_without_claims: bool = True  # מותר חישוב טהור בלי claims
    max_claims: int = 64

    def check_claims_payload(self, payload: Dict[str, Any]):
        # סכימה קשיחה ל-claims
        schema = {
            "type":"object",
            "required":["claims","evidence"],
            "properties":{
                "claims":{"type":"array","items":{
                    "type":"object",
                    "required":["id","text"],
                    "properties":{
                        "id":{"type":"string","minLength":1,"maxLength":128},
                        "text":{"type":"string","minLength":1}
                    }
                }},
                "evidence":{"type":"array","items":{
                    "type":"object",
                    "required":["sha256","ts","trust","url","sig_ok"],
                    "properties":{
                        "sha256":{"type":"string","minLength":64,"maxLength":64, "pattern":"^[0-9a-f]{64}$"},
                        "ts":{"type":"integer","minimum":0},
                        "trust":{"type":"number","minimum":0.0,"maximum":1.0},
                        "url":{"type":"string","minLength":1},
                        "sig_ok":{"type":"boolean"}
                    }
                }}
            }
        }
        validate_schema(payload, schema, "$.respond_payload")
        if len(payload["claims"]) > self.max_claims:
            raise ContractViolation("too_many_claims", detail={"max": self.max_claims})

    def _host_ok(self, url: str) -> bool:
        if not self.evidence.allowed_domains: return True
        host = urllib.parse.urlparse(url).hostname or ""
        return any(host.endswith(dom) for dom in self.evidence.allowed_domains)

    def enforce(self, text: str, claims: Optional[list], evidence: Optional[list]) -> None:
        if not text or not isinstance(text, str):
            raise ContractViolation("empty_text")

        if claims or evidence:
            self.check_claims_payload({"claims":claims or [], "evidence":evidence or []})

        if self.require_claims and not claims:
            # אולי זו תשובה מתמטית? נבדוק דגל
            if not self.allow_math_without_claims:
                raise PolicyDenied("claims_required", policy=self.as_dict())

        if self.require_evidence:
            if not evidence or len(evidence)==0:
                # חישוב טהור ללא מקור: אם מותר — אין בעיה
                if not (not claims and self.allow_math_without_claims):
                    raise PolicyDenied("evidence_required", policy=self.as_dict())
            else:
                now = int(time.time())
                for ev in evidence:
                    if self.evidence.require_signature and (not ev.get("sig_ok", False)):
                        raise PolicyDenied("evidence_signature_required", policy=self.as_dict())
                    if ev.get("trust", 0.0) < self.evidence.min_trust:
                        raise PolicyDenied("evidence_trust_too_low", policy=self.as_dict())
                    if (now - int(ev.get("ts",0))) > self.evidence.max_age_sec:
                        raise PolicyDenied("evidence_expired", policy=self.as_dict())
                    if not self._host_ok(ev.get("url","")):
                        raise PolicyDenied("evidence_domain_not_allowed", policy=self.as_dict())

    def as_dict(self) -> dict:
        return {
            "require_claims": self.require_claims,
            "require_evidence": self.require_evidence,
            "allow_math_without_claims": self.allow_math_without_claims,
            "max_claims": self.max_claims,
            "evidence":{
                "min_trust": self.evidence.min_trust,
                "max_age_sec": self.evidence.max_age_sec,
                "allowed_domains": self.evidence.allowed_domains,
                "require_signature": self.evidence.require_signature
            }
        }