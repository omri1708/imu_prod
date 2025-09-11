# tests/test_alerts_advanced_and_prbot.py
def test_alerts_advanced_values_present():
    txt=open("helm/umbrella/values.alerts.advanced.yaml","r",encoding="utf-8").read()
    assert "time_intervals:" in txt and "receiver: 'pager'" in txt

def test_pr_bot_workflow_exists():
    assert open(".github/workflows/pr-bot.yml","r",encoding="utf-8").read().startswith("name:")