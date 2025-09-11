# tests/test_k6_ws_publish_flow.py
def test_k6_ws_script_includes_publish_api():
    txt=open("helm/control-plane/templates/k6-configmap.yaml","r",encoding="utf-8").read()
    assert "/events/publish" in txt and "got at least 3 echoes" in txt