# tests/test_allowed_diffs_dashboard_json.py
import json
def test_allowed_diffs_dashboard_loads():
    j=json.load(open("monitoring/grafana/dashboards/imu_allowed_diffs.json","r",encoding="utf-8"))
    assert "panels" in j and any("imu_allowed_diffs_unexpected_total" in (t.get("expr","") or "") for p in j["panels"] for t in (p.get("targets") or []))
