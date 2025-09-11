# -*- coding: utf-8 -*-
from __future__ import annotations
import re, yaml, hashlib, os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from assurance.errors import ResourceRequired, ValidationFailed
from pathlib import Path

def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1024*1024), b""):
            h.update(ch)
    return h.hexdigest()

@dataclass
class ToolRule:
    name: str
    sha256: Optional[str] = None      # חתימת בינארי צפויה (הקשחה)
    args_regex: Optional[str] = None  # מגבלת ארגומנטים (רשות)
    allow_net: bool = False

@dataclass
class Policy:
    strict_fs: bool = True            # דורש bwrap; אחרת resource_required
    no_net_default: bool = True       # NET אסור כברירת מחדל
    cpu_seconds: int = 30
    mem_bytes: int = 256*1024*1024
    wall_seconds: int = 45
    open_files: int = 256
    allowed_tools: Dict[str, ToolRule] = None
    allow_env: List[str] = None       # מפתחות ENV מותרים

    @staticmethod
    def load(path: str) -> "Policy":
        y = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
        tools={}
        for t in (y.get("allowed_tools") or []):
            tools[t["name"]] = ToolRule(
                name=t["name"], sha256=t.get("sha256"), args_regex=t.get("args_regex"),
                allow_net=bool(t.get("allow_net", False)))
        return Policy(
            strict_fs=bool(y.get("strict_fs", True)),
            no_net_default=bool(y.get("no_net_default", True)),
            cpu_seconds=int(y.get("cpu_seconds", 30)),
            mem_bytes=int(y.get("mem_bytes", 256*1024*1024)),
            wall_seconds=int(y.get("wall_seconds", 45)),
            open_files=int(y.get("open_files", 256)),
            allowed_tools=tools,
            allow_env=list(y.get("allow_env", ["PATH","LANG","LC_ALL"]))
        )

    def tool_guard(self, exe_path: str, argv: List[str]):
        name = os.path.basename(exe_path)
        rule = self.allowed_tools.get(name)
        if not rule:
            raise ResourceRequired(f"tool_not_allowed:{name}", "add tool to executor/policy.yaml:allowed_tools")
        if rule.sha256:
            dig = sha256_file(exe_path)
            if dig != rule.sha256:
                raise ValidationFailed(f"tool_hash_mismatch:{name} expected {rule.sha256} got {dig}")
        if rule.args_regex:
            s = " ".join(argv[1:]) if len(argv) > 1 else ""
            if not re.fullmatch(rule.args_regex, s):
                raise ValidationFailed(f"args_not_allowed for {name}")
        return rule
