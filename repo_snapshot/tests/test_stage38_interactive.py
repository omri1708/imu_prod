# imu_repo/tests/test_stage38_interactive.py
from __future__ import annotations
import http.client, json
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline

def http_post(port: int, path: str, obj: dict) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3.0)
    data = json.dumps(obj).encode()
    conn.request("POST", path, body=data, headers={"Content-Type":"application/json"})
    r = conn.getresponse()
    r.read()
    return r.status

def run():
    schema = {
        "type":"object",
        "properties":{
            "perf":{"type":"object","properties":{"p95_ms":{"type":"number","maximum":1000}},"required":["p95_ms"]},
            "tests":{"type":"object"}
        },
        "required":["perf","tests"]
    }
    spec = BuildSpec(
        name="stage38_full",
        kind="web_service",
        language_pref=["python"],
        ports=[18181],
        endpoints={"/hello":"hello_json"},
        contracts=[Contract(name="svc_perf_ok", schema=schema)],
        evidence_requirements=["service_tests","perf_summary"]
    )
    summary = run_pipeline(spec, user_id="dana")
    ok = summary["tests"]["passed"] and summary["verify"]["ok"] and summary["rollout"]["approved"]

    # בדיקת POST /kv
    port = summary["tests"]["port"]
    st = http_post(port, "/kv", {"k":"x","v":"1"})
    ok = ok and (st==200)

    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())