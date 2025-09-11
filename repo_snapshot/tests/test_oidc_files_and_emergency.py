# tests/test_oidc_files_and_emergency.py

def test_values_oidc_and_gating_exist():
    assert open("helm/umbrella/values.oidc.yaml","r",encoding="utf-8").read().strip() != ""
    assert open("helm/umbrella/templates/gating-oidc-grafana.yaml","r",encoding="utf-8").read().startswith("{{- if")

def test_emergency_api_and_ui_exist():
    assert open("server/emergency_api.py","r",encoding="utf-8").read().startswith("# server/emergency_api.py")
    assert open("ui/emergency.html","r",encoding="utf-8").read().startswith("<!doctype html>")