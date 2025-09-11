# tests/test_loki_dashboard_file.py
import json
def test_policy_drilldown_dashboard_loads():
    j=json.load(open("monitoring/grafana/dashboards/imu_policy_drilldown.json","r",encoding="utf-8"))
    assert "panels" in j and any(p.get("type")=="logs" for p in j["panels"])