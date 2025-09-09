import os
from engine.prebuild.adapter_builder import ensure_capabilities

def test_adapter_builder_adoption(tmp_path, monkeypatch):
    os.chdir(tmp_path)
    spec = {"adapters": [{"kind":"k8s.apply"}]}
    ctx = {"user_id":"u","__policy__":{"capabilities":{"auto": True}}}
    out = ensure_capabilities(spec, ctx)
    assert isinstance(out, list)
    assert (tmp_path/"adapters").exists()
