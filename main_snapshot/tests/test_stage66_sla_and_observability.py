# imu_repo/tests/test_stage66_sla_and_observability.py
from __future__ import annotations
import os, json, time, tempfile, http.client
from typing import Dict, Any
from alerts.notifier import metrics_log, alert, ROOT as LOG_ROOT
from metrics.aggregate import aggregate_metrics
from sla.policy import SlaSpec, evaluate
from safe_progress.auto_rollout import decide, DEC_PROMOTE, DEC_HOLD, DEC_ROLLBACK
from observability.server import run as run_obs

def _reset_logs():
    try:
        os.makedirs(LOG_ROOT, exist_ok=True)
        for fn in ("metrics.jsonl","alerts.jsonl"):
            p = os.path.join(LOG_ROOT, fn)
            if os.path.exists(p): os.remove(p)
    except Exception: ...

def _gen(name: str, bucket: str, n: int, lat_ms: int, ok: bool=True, fail_every: int=0, gate_denied_every: int=0):
    os.environ["IMU_BUCKET"] = bucket
    for i in range(n):
        metrics_log(name, {"ok": ok, "latency_ms": lat_ms})
        if gate_denied_every and (i % gate_denied_every == 0):
            alert("evidence_gate_denied", severity="high", meta={})
        elif fail_every and (i % fail_every == 0):
            alert("handler_failure", severity="high", meta={})

def _http_get(host: str, port: int, path: str) -> int:
    c = http.client.HTTPConnection(host, port, timeout=1.5)
    c.request("GET", path)
    r = c.getresponse()
    r.read()
    c.close()
    return r.status

def test_sla_and_rollout_and_observability():
    _reset_logs()
    name="guarded_handler"
    # צור baseline מהיר ויציב יותר
    _gen(name, "baseline", n=200, lat_ms=40, ok=True, fail_every=0)
    # צור canary מעט איטי יותר + מספר gate_denied כדי לבחון החלטות
    _gen(name, "canary",   n=200, lat_ms=50, ok=True, gate_denied_every=51)

    # אגרגציה
    base = aggregate_metrics(name=name, bucket="baseline", window_s=3600)
    can  = aggregate_metrics(name=name, bucket="canary",   window_s=3600)
    assert base["count"]==200 and can["count"]==200

    # SLA: דורשים p95<=80ms, error_rate<=2%, gate_denied<=2%
    spec = SlaSpec("default", p95_ms=80.0, max_error_rate=0.02, max_gate_denied_rate=0.02, min_throughput_rps=0.0)
    sla_can = evaluate(can, spec)
    assert sla_can["ok"] is True  # למרות gate_denied נמוך יחסית, אמור לעבור

    # החלטת רולאאוט: canary איטי מבייסליין → not_worse=false → HOLD
    d = decide(window_s=3600, name=name, sla=spec, require_improvement=False)
    assert d["decision"] in (DEC_HOLD, DEC_ROLLBACK)

    # הרם Observability ובדוק שהוא מגיב
    t = run_obs()
    time.sleep(0.2)
    st1 = _http_get("127.0.0.1", 8799, "/metrics.json")
    st2 = _http_get("127.0.0.1", 8799, "/alerts.json")
    st3 = _http_get("127.0.0.1", 8799, "/")
    assert st1==200 and st2==200 and st3==200
    print("OK")
    return 0

if __name__=="__main__":
    raise SystemExit(test_sla_and_rollout_and_observability())