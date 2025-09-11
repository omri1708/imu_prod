# imu_repo/tests/test_provenance_and_policy.py
from __future__ import annotations
import os, json, time, hashlib, hmac
import shutil, tempfile
from engine.policy_compiler import compile_policy
from engine.respond_guard import ensure_proof_and_package, RespondBlocked
from provenance.store import CAS
from provenance.audit import AuditLog
from policy.policies import DEFAULT_POLICY
from policy.ttl import enforce_ttl


def test_cas_sign_and_verify():
    root = tempfile.mkdtemp()
    try:
        cas = CAS(root, b"secret")
        digest = cas.put(b"hello", "artifact", "team", {"note":"t"})
        assert cas.verify_meta(digest)
        assert cas.get(digest) == b"hello"
    finally:
        shutil.rmtree(root)


def test_ttl_cleanup():
    class MemIdx:
        def __init__(self):
            self.docs = {"claim":[], "evidence":[], "artifact":[], "session":[]}
        def iter_docs(self, kind):
            return list(self.docs[kind])
        def delete(self, doc_id):
            for k in self.docs:
                self.docs[k] = [x for x in self.docs[k] if x[0]!=doc_id]
    idx = MemIdx()
    now = time.time()
    idx.docs["claim"] = [("c1", now-DEFAULT_POLICY.ttl_seconds["claim"]-1)]
    removed = enforce_ttl(idx, DEFAULT_POLICY, now)
    assert removed["claim"] == 1


def _mk_sig(secret_hex: str, fields, e: dict, algo="sha256"):
    secret = bytes.fromhex(secret_hex)
    parts = []
    for f in fields:
        v = e
        for seg in f.split("."):
            v = v.get(seg) if isinstance(v, dict) else None
        if v is None: parts.append(b"")
        else: parts.append(str(v).encode("utf-8"))
    msg = b"\x1f".join(parts)
    dig = getattr(hashlib, algo)
    return hmac.new(secret, msg, dig).hexdigest()

def _ok_fetch(url: str, method: str):
    # מטא־דאטה "טרי"
    return (200, {"date":"Tue, 01 Jul 2025 12:00:00 GMT"}, b"")

def test_policy_compiler_and_signed_provenance(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    domain = {
      "trust_domains": {"example.com":3},
      "trusted_domains": ["example.com"],
      "min_distinct_sources": 1,
      "min_total_trust": 2,
      "signing_keys": {"k1":{"secret_hex":"aa"*32,"algo":"sha256"}},
      "signature_fresh_window_sec": 3600,
      "min_provenance_level": 2,  # דורש חתימה
      "p95_limits": {"plan": 200}
    }
    pol = compile_policy(json.dumps(domain))

    # evidence חתום על השדה url
    e = {"kind":"http","url":"https://api.example.com/x","signed_fields":["url"],"key_id":"k1"}
    e["sig"] = _mk_sig(domain["signing_keys"]["k1"]["secret_hex"], e["signed_fields"], e)

    claims = [{
        "id":"c1",
        "type":"latency",
        "text":"p95=120ms",
        "schema":{"type":"number","unit":"ms","min":0,"max":500},
        "value": 120,
        "evidence":[e],
        "consistency_group":"lat"
    }]

    out = ensure_proof_and_package(response_text="ok", claims=claims, policy=pol, http_fetcher=_ok_fetch)
    assert out["ok"]

def test_provenance_block_without_signature(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    domain = {
      "trust_domains": {"example.com":3},
      "trusted_domains": ["example.com"],
      "min_distinct_sources": 1,
      "min_total_trust": 1,
      "min_provenance_level": 3,  # דורש חתימה+fresh ts
      "signature_fresh_window_sec": 60
    }
    pol = compile_policy(json.dumps(domain))
    # אין חתימה/ts → צריך להיחסם
    claims = [{
        "id":"c2",
        "text":"val",
        "schema":{"type":"string","min_len":1},
        "value": "ok",
        "evidence":[{"kind":"http","url":"https://example.com/x"}]
    }]
    try:
        ensure_proof_and_package(response_text="x", claims=claims, policy=pol, http_fetcher=lambda u,m:(200,{"date":"Tue, 01 Jul 2025 12:00:00 GMT"},b""))
        assert False, "should block due to provenance"
    except RespondBlocked as e:
        assert "provenance_fail" in str(e)