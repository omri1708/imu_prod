# imu_repo/realtime/strict_sink.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import time, hashlib

class Reject(Exception):
    def __init__(self, reason: str, details: Dict[str, Any] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}

class StrictSink:
    """
    מסנן ריל־טיים: כל פלט היוצא ללקוח *חייב* לכלול claims+evidence, ולעבור בדיקות בסיסיות.
    אין מסרים "גלמיים". אין יציאה ללא הצדקה. אין "כמעט".
    """
    def __init__(self, policy: Dict[str, Any]):
        self.policy = dict(policy or {})
        perf = self.policy.get("perf_sla", {})
        self.p95_max = float(perf.get("latency_ms", {}).get("p95_max", 200.0))
        self.trust_min = float(self.policy.get("min_total_trust", 1.0))
        self.min_sources = int(self.policy.get("min_distinct_sources", 1))

    # ---- בדיקות עזר בסיסיות (ללא תלות חיצונית) ----
    @staticmethod
    def _distinct_sources(claim: Dict[str, Any]) -> int:
        seen = set()
        for ev in claim.get("evidence", []):
            if isinstance(ev, dict):
                src = ev.get("source") or ev.get("url") or ev.get("sha256") or ev.get("kind")
                if src: seen.add(str(src))
        return len(seen)

    @staticmethod
    def _claim_fingerprint(claim: Dict[str, Any]) -> str:
        import json
        b = json.dumps(claim, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(b).hexdigest()

    def _score_claim(self, claim: Dict[str, Any]) -> float:
        # ניקוד פשטני: מספר מקורות + בונוס ל-hash/sha (CAS) + בונוס ל-https
        score = 0.0
        distinct = self._distinct_sources(claim)
        score += distinct
        for ev in claim.get("evidence", []):
            if isinstance(ev, dict):
                if "sha256" in ev: score += 0.5
                url = ev.get("url") or ""
                if isinstance(url, str) and url.startswith("https://"): score += 0.25
        return score

    # ---- אימות bundle ----
    def verify_grounded(self, bundle: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        if "text" not in bundle or "claims" not in bundle:
            raise Reject("bundle_missing_fields", {"need": ["text", "claims"]})

        text = bundle["text"]
        claims = bundle["claims"]
        if not isinstance(text, str) or not isinstance(claims, list) or not claims:
            raise Reject("bad_types_or_empty_claims")

        total_score = 0.0
        worst_sources = 10 ** 9
        fps: List[str] = []
        for i, c in enumerate(claims):
            if not isinstance(c, dict):
                raise Reject("claim_not_object", {"index": i})
            if "type" not in c or "text" not in c:
                raise Reject("claim_missing_core_fields", {"index": i})
            ev = c.get("evidence", [])
            if not isinstance(ev, list) or not ev:
                raise Reject("claim_missing_evidence", {"index": i})
            ds = self._distinct_sources(c)
            if ds < self.min_sources:
                raise Reject("not_enough_sources", {"index": i, "have": ds, "need": self.min_sources})
            total_score += self._score_claim(c)
            worst_sources = min(worst_sources, ds)
            fps.append(self._claim_fingerprint(c))

        # ציון אמון מצטבר
        if total_score < self.trust_min:
            raise Reject("low_total_trust", {"got": total_score, "need": self.trust_min})

        return True, {"claim_fingerprints": fps, "min_distinct_sources": worst_sources, "trust_score": total_score}

    # ---- שער יציאה: כל הודעה החוצה עוברת כאן ----
    def guard_outbound(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """
        envelope = {"op": "...", "bundle": {...}}
        """
        op = envelope.get("op")
        bundle = envelope.get("bundle", {})

        # בקרה: הודעות בקרה מסוימות יכולות לעבור (ללא טקסט), אבל לא תוכן למשתמש.
        control_ops = {"control/ack", "control/error", "control/hello"}
        if op in control_ops:
            return envelope

        ok, meta = self.verify_grounded(bundle)
        # SLA עידון (הדגמתי hooks): ניתן לצרף מדדי זמן/latency לפני השליחה ולהשליך אם חורג
        # כאן איננו מודדים latency בפועל; אם policy כוללת p95 נדרשת – על המעלית לרשום.
        bundle["_verifier_meta"] = meta
        return {"op": op, "bundle": bundle}