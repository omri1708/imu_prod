# tests/test_values_prod_yaml.py
def test_prod_values_has_required_blocks():
    txt=open("helm/control-plane/values.production.yaml","r",encoding="utf-8").read()
    assert "namespace:" in txt and "images:" in txt and "ingress:" in txt