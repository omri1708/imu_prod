# tests/test_alert_pagers_values_and_gating.py
def test_values_alerts_pagers_file_present():
    assert open("helm/umbrella/values.alerts.pagers.yaml","r",encoding="utf-8").read().strip() != ""

def test_gating_alerts_template_has_pagerduty_opsgenie_checks():
    txt=open("helm/umbrella/templates/gating-alerts.yaml","r",encoding="utf-8").read()
    assert "PagerDuty receiver requires" in txt and "Opsgenie receiver requires" in txt