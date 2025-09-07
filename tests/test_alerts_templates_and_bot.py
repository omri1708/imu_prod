# tests/test_alerts_templates_and_bot.py
def test_alerts_values_and_templates_exist():
    assert open("helm/umbrella/values.alerts.yaml","r",encoding="utf-8").read().strip() != ""
    assert open("helm/umbrella/templates/alertmanager-configmap-templates.yaml","r",encoding="utf-8").read().startswith("apiVersion:")
    assert open("helm/umbrella/templates/gating-alerts.yaml","r",encoding="utf-8").read().startswith("{{- if")

def test_pr_bot_workflow_exists():
    assert open(".github/workflows/pr-bot.yml","r",encoding="utf-8").read().startswith("name:")