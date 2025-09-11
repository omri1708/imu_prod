# tests/test_grafana_dashboards_json.py
import json, os
def _load(p):
    with open(p,"r",encoding="utf-8") as f: return json.load(f)
def test_dashboards_loadable():
    base="monitoring/grafana/dashboards"
    assert os.path.exists(base)
    for fn in ("imu_api.json","imu_ws.json","imu_scheduler.json"):
        dash=_load(os.path.join(base,fn))
        assert "panels" in dash and isinstance(dash["panels"], list)