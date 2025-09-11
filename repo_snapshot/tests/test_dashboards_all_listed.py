# tests/test_dashboards_all_listed.py
def test_dashboards_configmap_lists_all():
    cm=open("helm/control-plane/templates/grafana-dashboards-cm.yaml","r",encoding="utf-8").read()
    for fn in (
        "imu_api.json","imu_ws.json","imu_scheduler.json",
        "imu_gatekeeper.json","imu_gate_trends.json",
        "imu_allowed_diffs.json","imu_policy_drilldown.json",
        "imu_slo.json","imu_kind_smoke_slo.json",
    ):
        assert fn in cm