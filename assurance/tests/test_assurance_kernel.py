# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from assurance.assurance import AssuranceKernel
from assurance.validators import schema_validator
from assurance.signing import Signer
from assurance.ops_registry import text_render_builder
from assurance.errors import RefusedNotGrounded, ValidationFailed
import json, shutil, os, pytest

def setup_module():
    if os.path.exists("./assurance_store_test"):
        shutil.rmtree("./assurance_store_test")

def test_commit_requires_evidence_and_validators():
    k = AssuranceKernel("./assurance_store_test")
    s = k.begin("x", "1.0.0", "demo")
    s.add_input_bytes("i", b"{}")
    s.set_builder(text_render_builder('{"ok":true}'))
    s.build()
    with pytest.raises(RefusedNotGrounded):
        s.commit(Signer())

def test_commit_with_evidence_and_validator_passes():
    k = AssuranceKernel("./assurance_store_test")
    s = k.begin("x", "1.0.0", "demo")
    s.add_input_bytes("i", b'{"title":"t","value":5}')
    s.add_evidence("source", "internal:unit", trust=0.9)
    s.attach_validator(schema_validator({"required":["title","value"],
                                         "properties":{"title":{"type":"string","minLength":1},
                                                       "value":{"type":"number","minimum":0,"maximum":10}}}))
    s.set_builder(text_render_builder(json.dumps({"title":"t","value":5})))
    s.build()
    rec = s.commit(Signer())
    assert "root" in rec and len(rec["root"]) == 64

def teardown_module():
    # deliberately keep store for inspection; uncomment if you want cleanup
    # shutil.rmtree("./assurance_store_test", ignore_errors=True)
    pass
