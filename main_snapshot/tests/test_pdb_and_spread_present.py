# tests/test_pdb_and_spread_present.py

def test_pdb_and_spread_snippets_present():
    assert open("helm/control-plane/templates/pdb.yaml","r",encoding="utf-8").read().startswith("apiVersion:")
    a=open("helm/control-plane/templates/deployment-api.yaml","r",encoding="utf-8").read()
    w=open("helm/control-plane/templates/deployment-ws.yaml","r",encoding="utf-8").read()
    assert "topologySpreadConstraints" in a and "topologySpreadConstraints" in w