# imu_repo/provenance/provenance.py
from __future__ import annotations
import json, time, os
from typing import Dict, Any, List, Optional
from provenance.cas import CAS
from security.signing import sign_manifest, verify_manifest
from provenance.trust_registry import TrustRegistry

class ProvenanceError(Exception): ...
class TrustError(ProvenanceError): ...

def now_ts() -> int: return int(time.time())

def normalize_evidence(ev: Dict[str,Any], tr: TrustRegistry) -> Dict[str,Any]:
    out = {
        "kind": ev.get("kind","unknown"),
        "payload": ev.get("payload", {}),
        "source_url": ev.get("source_url",""),
        "ts": int(ev.get("ts", now_ts())),
        "ttl_s": int(ev.get("ttl_s", 3600)),
    }
    base_trust = float(ev.get("trust", tr.trust_for(out["source_url"])))
    # דעיכה קלה בזמן (עד 0.2 בחודש):
    age_s = max(0, now_ts() - out["ts"])
    decay = min(0.2, age_s / (30*24*3600) * 0.2)
    out["trust"] = max(0.0, min(1.0, base_trust - decay))
    return out

def aggregate_trust(evidences: List[Dict[str,Any]]) -> float:
    if not evidences: return 0.0
    by_src = {}
    for e in evidences:
        src = e.get("source_url","")
        by_src.setdefault(src, []).append(e)
    diversity_bonus = min(0.1, 0.03 * (len(by_src)-1))
    base = sum(e["trust"] for e in evidences)/len(evidences)
    return min(1.0, base + diversity_bonus)

def evidence_expired(e: Dict[str,Any]) -> bool:
    return (now_ts() - int(e.get("ts", now_ts()))) > int(e.get("ttl_s", 3600))

class ProvenanceStore:
    def __init__(self, cas: CAS, *, min_trust: float=0.75, trust_registry_path: Optional[str]=None):
        self.cas = cas
        self.min_trust = float(min_trust)
        self.registry = TrustRegistry(trust_registry_path or os.environ.get("IMU_TRUST_PATH","/mnt/data/.imu_trust.json"))

    def ingest_evidences(self, evidences: List[Dict[str,Any]]) -> str:
        norm = [normalize_evidence(e, self.registry) for e in evidences if not evidence_expired(e)]
        evdoc = {"_type":"evidences","items": norm, "agg_trust": aggregate_trust(norm), "ts": now_ts()}
        # חתימת evidences manifest (שרשרת שלמה)
        signed_evs = sign_manifest(evdoc, key_id=os.environ.get("IMU_EVIDENCE_KEY","default"))
        sha_signed = self.cas.put(json.dumps(signed_evs, sort_keys=True).encode("utf-8"),
                                  kind="evidences+signature", mime="application/json")
        self.cas.link(f"evidences/{evdoc['ts']}", sha_signed, note="signed evidences")
        return sha_signed

    def _load_signed_evidences(self, ev_sha: str) -> Dict[str,Any]:
        signed = json.loads(self.cas.get(ev_sha).decode("utf-8"))
        # אימות חתימה על מסמך הראיות
        verify_manifest(signed)
        return signed["payload"]

    def attach_artifact(self, data: bytes, meta: Dict[str,Any], *,
                        evidences_sha: str, key_id: str="default") -> Dict[str,Any]:
        blob_sha = self.cas.put(data, kind=meta.get("kind","artifact"),
                                mime=meta.get("mime","application/octet-stream"),
                                extra_meta=meta)
        evdoc = self._load_signed_evidences(evidences_sha)
        agg = float(evdoc.get("agg_trust",0.0))
        if agg < self.min_trust:
            raise TrustError(f"aggregate trust {agg:.2f} < min_trust {self.min_trust:.2f}")
        manifest = {
            "artifact_sha256": blob_sha,
            "artifact_meta": meta,
            "evidences_sha256": evidences_sha,  # מצביע למסמך ראיות חתום
            "agg_trust": agg,
            "created_ts": now_ts()
        }
        signed = sign_manifest(manifest, key_id=key_id)
        man_sha = self.cas.put(json.dumps(signed, sort_keys=True).encode("utf-8"),
                               kind="manifest", mime="application/json")
        self.cas.link(f"artifact/{blob_sha}", blob_sha, note="artifact")
        self.cas.link(f"manifest/{blob_sha}", man_sha, note="manifest")
        self.cas.link("latest/artifact", blob_sha, note="latest")
        self.cas.link("latest/manifest", man_sha, note="latest")
        return {"artifact_sha": blob_sha, "manifest_sha": man_sha, "agg_trust": agg}

    def verify_chain(self, manifest_sha: str) -> Dict[str,Any]:
        signed = json.loads(self.cas.get(manifest_sha).decode("utf-8"))
        verify_manifest(signed)
        payload = signed["payload"]
        art_sha = payload["artifact_sha256"]
        ev_sha  = payload["evidences_sha256"]
        self.cas.verify_blob(art_sha)
        self.cas.verify_blob(ev_sha)
        evdoc = self._load_signed_evidences(ev_sha)
        agg = float(evdoc.get("agg_trust",0.0))
        if agg < self.min_trust:
            raise TrustError(f"aggregate trust {agg:.2f} < min_trust {self.min_trust:.2f}")
        return {"ok": True, "artifact_sha": art_sha, "evidences_sha": ev_sha, "agg_trust": agg, "manifest": signed}