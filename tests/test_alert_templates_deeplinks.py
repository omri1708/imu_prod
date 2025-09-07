# tests/test_alert_templates_deeplinks.py
def test_slack_template_contains_deeplinks():
    txt=open("helm/umbrella/templates/alertmanager-configmap-templates.yaml","r",encoding="utf-8").read()
    assert "grafanaUrl" in txt and "sloUid" in txt and "gateUid" in txt