# tests/test_grounded_enforced.py
# -*- coding: utf-8 -*-
import pytest
from engine.respond import GroundedResponder

def test_block_without_evidence():
    gr = GroundedResponder(trust_threshold=0.6)
    with pytest.raises(RuntimeError):
        gr.respond({"__claims__": []}, "nope")

def test_allow_with_valid_evidence(tmp_path, monkeypatch):
    # נכניס meta ל-CAS
    from provenance.store import CASStore
    cas = CASStore(str(tmp_path/"cas"), str(tmp_path/"keys"))
    meta = cas.put_bytes(b"hello", sign=True, url="https://a", trust=0.9, not_after_days=1)
    gr = GroundedResponder(trust_threshold=0.6)
    out = gr.respond({"__claims__":[{"sha256": meta.sha256}]}, "ok")
    assert out["ok"] and out["text"] == "ok"