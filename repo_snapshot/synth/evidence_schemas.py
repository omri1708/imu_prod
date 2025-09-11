# imu_repo/synth/evidence_schemas.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List

def _req(d: Dict[str,Any], keys: List[str]) -> Tuple[bool, str]:
    for k in keys:
        if k not in d:
            return False, f"missing:{k}"
    return True, "ok"

def schema_spec(spec: Dict[str,Any]) -> Tuple[bool, str]:
    """
    סכימה פשוטה ל-Spec של משימה.
    """
    ok, why = _req(spec, ["name","goal"])
    if not ok: return ok, why
    if not isinstance(spec["name"], str) or not spec["name"]:
        return False, "name:not_str_or_empty"
    if not isinstance(spec["goal"], str) or not spec["goal"]:
        return False, "goal:not_str_or_empty"
    return True, "ok"

def schema_plan(plan: Dict[str,Any]) -> Tuple[bool, str]:
    ok, why = _req(plan, ["steps"])
    if not ok: return ok, why
    if not isinstance(plan["steps"], list) or not plan["steps"]:
        return False, "steps:not_list_or_empty"
    return True, "ok"

def schema_generate(gen: Dict[str,Any]) -> Tuple[bool, str]:
    ok, why = _req(gen, ["language","code"])
    if not ok: return ok, why
    if gen["language"] not in ("python","js","go","rust","csharp","java"):
        return False, "language:unsupported"
    if not isinstance(gen["code"], str) or not gen["code"]:
        return False, "code:empty"
    return True, "ok"

def schema_test(res: Dict[str,Any]) -> Tuple[bool, str]:
    ok, why = _req(res, ["unit","integration"])
    if not ok: return ok, why
    if not isinstance(res["unit"], dict) or not isinstance(res["integration"], dict):
        return False, "test:bad_types"
    if not res["unit"].get("passed", False):
        return False, "unit:failed"
    if not res["integration"].get("passed", False):
        return False, "integration:failed"
    return True, "ok"

def schema_verify(v: Dict[str,Any]) -> Tuple[bool,str]:
    ok, why = _req(v, ["static_ok","style_ok"])
    if not ok: return ok, why
    if not (bool(v["static_ok"]) and bool(v["style_ok"])):
        return False, "verify:not_all_ok"
    return True, "ok"

def schema_package(pkg: Dict[str,Any]) -> Tuple[bool,str]:
    ok, why = _req(pkg, ["artifact_name","artifact_text","lang"])
    if not ok: return ok, why
    if not isinstance(pkg["artifact_text"], str) or not pkg["artifact_text"]:
        return False, "artifact_text:empty"
    return True, "ok"