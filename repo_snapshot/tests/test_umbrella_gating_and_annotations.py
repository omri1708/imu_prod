# tests/test_umbrella_gating_and_annotations.py
def test_umbrella_chart_has_dependencies_and_gating():
    chart = open("helm/umbrella/Chart.yaml","r",encoding="utf-8").read()
    assert "ingress-nginx" in chart and "cert-manager" in chart and "external-dns" in chart
    vals = open("helm/umbrella/values.yaml","r",encoding="utf-8").read()
    assert "gating:" in vals and "allowedIngressClasses" in vals

def test_umbrella_prod_app_has_image_updater_annotations():
    app = open("argocd/apps/children/umbrella-prod.yaml","r",encoding="utf-8").read()
    assert "argocd-image-updater.argoproj.io/image-list" in app
