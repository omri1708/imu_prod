# tests/test_pipeline_with_policy.py
# -*- coding: utf-8 -*-
from engine.synthesis_pipeline import run_pipeline
from governance.user_policy import ensure_user, restrict_domains

SPEC = """
name: demo
targets:
  - path: app/main.txt
    content: Hello
"""

def test_pipeline_power_user():
    user = "power_user"
    ensure_user(user)
    out = run_pipeline(user, SPEC)
    assert out["ok"] is True
    assert out["pkg"]

def test_pipeline_strict_domain_blocks():
    user = "strict_org"
    ensure_user(user)
    restrict_domains(user, ["corp.example"])  # generate ייצור evidences ל-example.com → ייחסם
    out = run_pipeline(user, SPEC)
    assert out["ok"] is False
    assert out["stage"] in ("canary","verify","test","canary","rollout","generate","parse","plan","verify")