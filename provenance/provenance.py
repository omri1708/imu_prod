from __future__ import annotations
import json, time, hashlib, hmac, os
from typing import Dict, Any, List, Optional, Tuple
from provenance.cas import CAS
from security.signing import sign_manifest, verify_manifest

class ProvenanceError(Exception): ...
class TrustError(ProvenanceError): ...

TRUST_TIERS = {
    # מקור -> ציון בסיס (0..1). ניתן לעדכן בקובץ הגדרות חיצוני בהמשך.
    "imu://ui/sandbox": 0.90,
    "imu://ui/table":   0.94,
    "http://": 0.50, "https://": 0.70,  # דיפולטים למקורות רשת כלליים
}

def trust_of_source(url: str) -> float:
    for k,v in TRUST_TIERS.items():
        if url.startswith(k): return v
    return 0.5

def now_ts() -> int: return int(time.time())

def normalize_evidence(ev: Dict[str,Any]) -> Dict[str,Any]:
    """
    evidence: {kind, payload, source_url, trust, ttl_s, ts?}
    """
    out = {
        "kind": ev.get("kind","unknown"),
        "payload": ev.get("payload", {}),
        "source_url": ev.get("source_url",""),
        "ts": int(ev.get("ts", now_ts())),
        "ttl_s": int(ev.get("ttl_s", 3600)),
    }
    base_trust = float(ev.get("trust", trust_of_source(out["source_url"])))
    # הנחת דעיכה קלה בזמן:
    age_s = max(0, now_ts() - out["ts"])
    decay = min(0.2, age_s / (30*24*3600) * 0.2)  # עד 0.2 הורדה בחודש
    out["trust"] = max(0.0, min(1.0, base_trust - decay))
    return out

def aggregate_trust(evidences: List[Dict[str,Any]]) -> float:
    """
    ממוצע משוקלל לפי אמון־מקור ו־diversity: מקורות שונים ↗
    """
    if not evidences: return 0.0
    by_src = {}
    for e in evidences:
        src = e.get("source_url","")
        by_src.setdefault(src, []).append(e)
    diversity_bonus = min(0.1, 0.03 * (len(by_src)-1))  # עד +0.1
    base = sum(e["trust"] for e in evidences)/len(evidences)
    return min(1.0, base + diversity_bonus)

def evidence_expired(e: Dict[str,Any]) -> bool:
    return (now_ts() - int(e.get("ts", now_ts()))) > int(e.get("ttl_s", 3600))

class ProvenanceStore:
    """
    שומר ארטיפקטים + רישום ראיות + Manifest חתום + קישורים נוחים.
    """
    def __init__(self, cas: CAS, *, min_trust: float=0.75):
        self.cas = cas
        self.min_trust = float(min_trust)

    def ingest_evidences(self, evidences: List[Dict[str,Any]]) -> str:
        norm = [normalize_evidence(e) for e in evidences if not evidence_expired(e)]
        doc = {"_type":"evidences","items": norm, "agg_trust": aggregate_trust(norm), "ts": now_ts()}
        sha = self.cas.put(json.dumps(doc, sort_keys=True).encode("utf-8"),
                           kind="evidences", mime="application/json")
        self.cas.link(f"evidences/{doc['ts']}", sha, note="snapshot")
        return sha

    def attach_artifact(self, data: bytes, meta: Dict[str,Any], *,
                        evidences_sha: str, key_id: str="default") -> Dict[str,Any]:
        # 1) שים את ה־blob (HTML למשל)
        blob_sha = self.cas.put(data, kind=meta.get("kind","artifact"), mime=meta.get("mime","text/html"), extra_meta=meta)
        # 2) משוך evidences
        evdoc = json.loads(self.cas.get(evidences_sha).decode("utf-8"))
        agg = float(evdoc.get("agg_trust",0.0))
        if agg < self.min_trust:
            raise TrustError(f"aggregate trust {agg:.2f} < min_trust {self.min_trust:.2f}")
        # 3) בנה manifest חתום
        manifest = {
            "artifact_sha256": blob_sha,
            "artifact_meta": meta,
            "evidences_sha256": evidences_sha,
            "agg_trust": agg,
            "created_ts": now_ts()
        }
        signed = sign_manifest(manifest, key_id=key_id)
        man_sha = self.cas.put(json.dumps(signed, sort_keys=True).encode("utf-8"),
                               kind="manifest", mime="application/json")
        # 4) קישוריות נוחה
        self.cas.link(f"artifact/{blob_sha}", blob_sha, note="artifact")
        self.cas.link(f"manifest/{blob_sha}", man_sha, note="manifest")
        self.cas.link("latest/artifact", blob_sha, note="latest")
        self.cas.link("latest/manifest", man_sha, note="latest")
        return {"artifact_sha": blob_sha, "manifest_sha": man_sha, "agg_trust": agg}

    def verify_chain(self, manifest_sha: str) -> Dict[str,Any]:
        s = json.loads(self.cas.get(manifest_sha).decode("utf-8"))
        # יאמת חתימה
        verify_manifest(s)
        art_sha = s["artifact_sha256"]
        ev_sha  = s["evidences_sha256"]
        # אימות אינטגריטי
        self.cas.verify_blob(art_sha)
        self.cas.verify_blob(ev_sha)
        # min trust
        evdoc = json.loads(self.cas.get(ev_sha).decode("utf-8"))
        agg = float(evdoc.get("agg_trust",0.0))
        if agg < self.min_trust:
            raise TrustError(f"aggregate trust {agg:.2f} < min_trust {self.min_trust:.2f}")
        return {"ok": True, "artifact_sha": art_sha, "evidences_sha": ev_sha, "agg_trust": agg, "signed": s}