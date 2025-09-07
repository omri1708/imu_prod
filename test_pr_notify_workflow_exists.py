# tests/test_pr_notify_workflow_exists.py
def test_pr_notify_workflow_exists():
    assert open(".github/workflows/pr-notify-slack-teams.yml","r",encoding="utf-8").read().startswith("name:")