# tests/test_argocd_apps_yaml.py
def test_argocd_manifests_exist():
    assert open("argocd/apps/app-of-apps.yaml","r",encoding="utf-8").read().startswith("apiVersion:")
    assert open("argocd/apps/children/control-plane-dev.yaml","r",encoding="utf-8").read().startswith("apiVersion:")
    assert open("argocd/apps/children/control-plane-prod.yaml","r",encoding="utf-8").read().startswith("apiVersion:")