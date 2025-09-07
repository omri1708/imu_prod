# tests/test_umbrella_chart_and_argocd.py
def test_umbrella_chart_and_values_exist():
    assert open("helm/umbrella/Chart.yaml","r",encoding="utf-8").read().startswith("apiVersion:")
    assert open("helm/umbrella/values.yaml","r",encoding="utf-8").read().strip() != ""
    assert open("helm/umbrella/values.prod.yaml","r",encoding="utf-8").read().strip() != ""

def test_argocd_projects_and_apps_exist():
    for p in (
        "argocd/projects/dev.yaml",
        "argocd/projects/staging.yaml",
        "argocd/projects/prod.yaml",
        "argocd/apps/children/umbrella-dev.yaml",
        "argocd/apps/children/umbrella-staging.yaml",
        "argocd/apps/children/umbrella-prod.yaml",
        "argocd/rbac/argocd-rbac-cm.yaml",
    ):
        assert open(p,"r",encoding="utf-8").read().startswith("apiVersion:")