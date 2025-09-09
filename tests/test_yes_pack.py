import os, json, time, shutil, tempfile
from pathlib import Path

# 1) ensure_tools minimal
from engine.prebuild.tool_acquisition import ensure_tools

def test_ensure_tools_noop(tmp_path):
    spec = {"tools_required": []}
    out = ensure_tools(spec, {"__policy__": {}})
    assert isinstance(out, list)

# 2) adapter_builder stub
from engine.prebuild.adapter_builder import ensure_capabilities

def test_adapter_builder_stub(tmp_path, monkeypatch):
    cwd = os.getcwd(); os.chdir(tmp_path)
    try:
        spec = {"adapters": [{"kind":"k8s.apply"}]}
        out = ensure_capabilities(spec, {"user_id":"u"})
        assert out and out[0]["ok"] in (True, False)
        # נוצר קובץ stub?
        assert (tmp_path/"adapters").exists()
    finally:
        os.chdir(cwd)

# 3) cache get/put/near-hit
from engine.llm.cache import LLMCache

def test_cache_put_get_nearhit(tmp_path):
    c = LLMCache(root=str(tmp_path/"cache"))
    key = c.make_key(model="m", system_v="1", template_v="1", tools_set="", user_text_norm="hello", ctx_ids="", persona_v="1", policy_v="1")
    c.put(key, model="m", payload={"content":"A","meta":{}, "_user_text_norm":"hello"}, ttl_s=5)
    ok, ent = c.get(key)
    assert ok and ent.payload["content"] == "A"
    nh = c.near_hit(query="hello there", model="m")
    assert nh and nh[0].payload["content"] == "A"

# 4) merkle audit verify
from audit.merkle_log import MerkleAudit

def test_merkle_audit(tmp_path):
    a = MerkleAudit(str(tmp_path/"audit"))
    a.append("progress", {"v":1}); a.append("progress", {"v":2})
    assert a.verify() is True

# 5) fault bisection
from engine.debug.fault_localizer import bisect_steps

def test_bisection():
    steps = []
    for i in range(10):
        def _mk(i=i):
            return {"ok": i < 6}
        steps.append(_mk)
    idx, info = bisect_steps(steps)
    assert idx == 6

# 6) hypothesis lab offline
from engine.research.hypothesis_lab import run_offline_experiments

def test_offline_lab():
    cfgs = [{"model":"gpt-4o-mini"}, {"model":"gpt-4o"}]
    rep = run_offline_experiments(cfgs, runner=lambda c: {"p95_ms": 1200.0 if c["model"]=="gpt-4o-mini" else 1600.0})
    assert rep["ok"] is True and rep["report"]["n"] == 2