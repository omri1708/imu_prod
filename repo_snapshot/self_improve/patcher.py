# imu_repo/self_improve/patcher.py
from __future__ import annotations
from typing import Dict, Any, List
from engine.config import load_config, save_config
from self_improve.fix_plan import FixPlan, FixAction

def _get_ref(cfg: Dict[str, Any], path: List[str]) -> tuple[Dict[str, Any], str]:
    cur = cfg
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    return cur, path[-1]

def apply_action(cfg: Dict[str,Any], act: FixAction)->None:
    parent, key = _get_ref(cfg, act.path)
    old = parent.get(key)
    if act.op == "set":
        parent[key] = act.value
    elif act.op == "inc":
        if isinstance(old, (int,float)):
            parent[key] = type(old)(old + act.value)
        else:
            parent[key] = act.value
    elif act.op == "dec":
        if isinstance(old, (int,float)):
            parent[key] = type(old)(old - act.value)
        else:
            parent[key] = act.value

def apply_plan(plan: FixPlan)->Dict[str,Any]:
    cfg = load_config()
    for a in plan.actions:
        apply_action(cfg, a)
    save_config(cfg)
    return cfg

def apply_all(plans: List[FixPlan])->Dict[str,Any]:
    cfg = load_config()
    for p in plans:
        for a in p.actions:
            apply_action(cfg, a)
    save_config(cfg)
    return cfg