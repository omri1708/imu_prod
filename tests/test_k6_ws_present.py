# tests/test_k6_ws_present.py
def test_k6_ws_script_contains_ws_logic():
    txt=open("helm/control-plane/templates/k6-configmap.yaml","r",encoding="utf-8").read()
    assert "import ws from 'k6/ws'" in txt and "WS_URL" in txt