# tests/test_ws_synthetic_script_and_alerts.py
import os
def test_ws_synthetic_script_exists():
    assert os.path.exists("scripts/ws_synthetic_ci.py")
def test_prometheusrule_alerts_exists():
    assert open("helm/control-plane/templates/prometheusrule-alerts.yaml","r",encoding="utf-8").read().startswith("apiVersion:")
