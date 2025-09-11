# tests/test_gatekeeper_dashboard_json.py
import json
def test_gatekeeper_dashboard_json_loads():
    j=json.load(open("monitoring/grafana/dashboards/imu_gatekeeper.json","r",encoding="utf-8"))
    assert "panels" in j and any("gatekeeper_violations" in (t.get("expr","") or "") for p in j["panels"] for t in (p.get("targets") or []))