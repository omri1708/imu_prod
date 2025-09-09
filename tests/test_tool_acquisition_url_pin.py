import io, json
from engine.prebuild.tool_acquisition import _install, ToolSpec

def test_url_blocked_domain(monkeypatch):
    t = ToolSpec(name="x", pkg="https://bad.com/file.bin", manager="url", sha256="00", spdx="MIT")
    res = _install(t)
    assert res.ok is False and "domain_not_allowed" in res.err

def test_url_ok_with_hash(monkeypatch, tmp_path):
    data = b"abc123"
    class _Resp:
        def read(self): return data
        def __enter__(self): return self
        def __exit__(self, *a): pass
    def fake_urlopen(url, timeout=30): return _Resp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    from hashlib import sha256
    t = ToolSpec(name="xbin", pkg="https://github.com/file.bin", manager="url",
                 sha256=sha256(data).hexdigest(), spdx="MIT")
    res = _install(t)
    assert res.ok is True and res.evidence_id
