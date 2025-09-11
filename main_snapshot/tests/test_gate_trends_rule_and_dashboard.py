# tests/test_gate_trends_rule_and_dashboard.py
def test_prometheusrule_and_trends_dashboard_exist():
    assert open("helm/control-plane/templates/prometheusrule-gatekeeper.yaml","r",encoding="utf-8").read().startswith("apiVersion:")
    assert open("monitoring/grafana/dashboards/imu_gate_trends.json","r",encoding="utf-8").read().startswith("{")