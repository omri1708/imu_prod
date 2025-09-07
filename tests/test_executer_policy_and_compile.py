# test/test_executer_policy_and_compile.py
from __future__ import annotations
import asyncio, json, os, pytest
from executor.policy import Policy
from executor.sandbox import SandboxExecutor, Limits
from assurance.errors import ResourceRequired, ValidationFailed

def test_policy_load_and_tools():
    p = Policy.load("./executor/policy.yaml")
    assert isinstance(p.allowed_tools, dict)
    # gcc רשאי (גם בלי sha לצורך הדוגמה)
    assert "gcc" in p.allowed_tools

@pytest.mark.asyncio
async def test_echo_native_or_unshare():
    ex = SandboxExecutor("./executor/policy.yaml", "./sbx_test")
    try:
        rc, out = await ex.run(["echo", "hi"], inputs={}, allow_write=["out"], limits=Limits(no_net=True))
        assert rc == 0 and b"hi" in out
    except ResourceRequired as e:
        # ייתכן שאין unshare/bwrap בסביבה — זה תקין לבדיקת משאבים
        assert "tool:" in str(e)

@pytest.mark.asyncio
async def test_compile_c_or_resource_required():
    # smoke הרצה — אם אין gcc/bwrap, נוודא שהמערכת לא “ממציאה”
    from integration.compile_c import compile_and_run_c
    try:
        r = await compile_and_run_c()
        assert r.get("ok") is True and "hello-from-gcc" in r.get("stdout","")
    except ResourceRequired as e:
        assert "tool:" in str(e) or "net_namespace" in str(e)
