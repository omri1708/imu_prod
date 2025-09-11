# tests/test_one_button_and_slo_dashboard.py
import os, json, stat

def test_one_button_script_exists():
    p="scripts/one_button_platform.sh"
    assert os.path.exists(p)
    assert os.stat(p).st_mode & stat.S_IXUSR

def test_slo_dashboard_loads():
    j=json.load(open("monitoring/grafana/dashboards/imu_slo.json","r",encoding="utf-8"))
    assert "panels" in j and any("imu_helm_test_pass" in (t.get("expr","") or "") for p in j["panels"] for t in (p.get("targets") or []))