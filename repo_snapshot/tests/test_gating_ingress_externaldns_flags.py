# tests/test_gating_ingress_externaldns_flags.py
def test_gating_values_flags_present():
    v=open("helm/umbrella/values.yaml","r",encoding="utf-8").read()
    assert "requireExternalDNSForIngress" in v
    g=open("helm/umbrella/templates/gating.yaml","r",encoding="utf-8").read()
    assert "externalDNS is disabled" in g