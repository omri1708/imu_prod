# policy/lint.py
# Lint בסיסי ל-policy_rules.yaml כדי להגן מפני תקלות שכיחות.
from __future__ import annotations
from typing import Dict, Any, List
import yaml

class PolicyLintError(ValueError): ...

MANDATORY_TOP = {"user_policies"}

def lint_yaml_text(yaml_text: str) -> Dict[str, Any]:
    try:
        cfg = yaml.safe_load(yaml_text) or {}
    except Exception as e:
        raise PolicyLintError(f"yaml parse error: {e}")
    # מפתחות עליונים
    for k in MANDATORY_TOP:
        if k not in cfg:
            raise PolicyLintError(f"missing top-level key: {k}")
    ups = cfg.get("user_policies", {})
    if not isinstance(ups, dict):
        raise PolicyLintError("user_policies must be a mapping")
    # ולידציות מינימליות למפתחות תוכן
    for uid, spec in ups.items():
        if not isinstance(spec, dict):
            raise PolicyLintError(f"user_policies.{uid} must be a mapping")
        if spec.get("default_net") not in ("deny","allow"):
            raise PolicyLintError(f"default_net invalid for {uid}")
        if spec.get("default_fs") not in ("deny","allow"):
            raise PolicyLintError(f"default_fs invalid for {uid}")
        # net_allow entries
        for i,rule in enumerate(spec.get("net_allow", [])):
            if "host" not in rule or "ports" not in rule:
                raise PolicyLintError(f"net_allow[{i}] missing host/ports for {uid}")
            if not isinstance(rule["ports"], list) or not all(isinstance(p,int) for p in rule["ports"]):
                raise PolicyLintError(f"net_allow[{i}].ports must be int list for {uid}")
        for i,rule in enumerate(spec.get("fs_allow", [])):
            if "path" not in rule or "mode" not in rule:
                raise PolicyLintError(f"fs_allow[{i}] missing path/mode for {uid}")
            if rule["mode"] not in ("ro","rw"):
                raise PolicyLintError(f"fs_allow[{i}].mode invalid for {uid}")
    return cfg