# imu_repo/tests/test_stage58_grounding.py
from __future__ import annotations
import time
from grounded.evidence_store import EvidenceStore
from engine.gates.grounding_gate import GroundingGate
from engine.runtime_bridge import apply_runtime_gates

def build_store():
    st = EvidenceStore()
    # עדות תקפה מלפני רגע, דומיין example.com
    sha_ok = st.put(source_url="https://example.com/doc1",
                    content="The capital of France is Paris.",
                    content_type="text/plain",
                    ttl_s=3600)  # שעה
    # עדות פגה (נגדיר stored_at בעבר)
    old_ts = time.time() - 10_000
    sha_old = st.put(source_url="https://example.com/old",
                     content="Outdated note",
                     content_type="text/plain",
                     ttl_s=60,
                     stored_at=old_ts)
    # עדות מדומיין לא מורשה
    sha_bad = st.put(source_url="https://untrusted.bad/news",
                     content="Unknown source",
                     content_type="text/plain",
                     ttl_s=3600)
    return st, sha_ok, sha_old, sha_bad

def bundle_good(sha_ok: str):
    return {
        "text": "Paris is the capital of France.",
        "claims": [
            {"id":"c1",
             "statement":"capital(france)=paris",
             "evidence":[sha_ok],
             "schema":{"type":"string","value":"Paris","pattern":"^[A-Z].+"}}
        ]
    }

def bundle_missing_claims():
    return {"text":"No claims here","claims":[]}

def bundle_expired(sha_old: str):
    return {
        "text":"Old statement",
        "claims":[{"id":"c2","statement":"old","evidence":[sha_old]}]
    }

def bundle_bad_domain(sha_bad: str):
    return {
        "text":"From bad domain",
        "claims":[{"id":"c3","statement":"x","evidence":[sha_bad]}]
    }

def run():
    st, ok_sha, old_sha, bad_sha = build_store()

    # 1) חבילת תשובה טובה — אמורה לעבור
    gate = GroundingGate(allowed_domains=["example.com"], require_signature=True, min_good_evidence=1)
    res1 = gate.check(bundle_good(ok_sha))
    ok1 = res1["ok"]

    # 2) ללא claims — חייב להיכשל
    res2 = gate.check(bundle_missing_claims())
    ok2 = (not res2["ok"] and res2["violations"][0][0]=="no_claims")

    # 3) עדות פגה — נכשל
    res3 = gate.check(bundle_expired(old_sha))
    ok3 = (not res3["ok"])

    # 4) דומיין לא מורשה — נכשל
    res4 = gate.check(bundle_bad_domain(bad_sha))
    ok4 = (not res4["ok"])

    # 5) אינטגרציה דרך runtime_bridge: הפעל grounding כ-Gate מערכתי
    extras = {"grounding":{"allowed_domains":["example.com"],"require_signature":True,"min_good_evidence":1}}
    out = apply_runtime_gates(extras, bundle=bundle_good(ok_sha))
    ok5 = (out.get("grounding",{}).get("ok") is True)

    ok_all = all([ok1, ok2, ok3, ok4, ok5])
    print("OK" if ok_all else f"FAIL res1={res1} res2={res2} res3={res3} res4={res4} out={out}")
    return 0 if ok_all else 1

if __name__=="__main__":
    raise SystemExit(run())