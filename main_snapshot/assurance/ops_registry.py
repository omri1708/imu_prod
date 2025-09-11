# -*- coding: utf-8 -*-
from __future__ import annotations
import shutil, subprocess, os
from typing import List
from .errors import ResourceRequired

def which_or_require(tool: str, install_hint: str) -> str:
    p = shutil.which(tool)
    if not p:
        raise ResourceRequired(f"tool:{tool}", how_to_get=install_hint)
    return p

def shell_builder(cmd: List[str]) -> callable:
    """
    Returns a builder(Session)->[digests] that runs a safe, explicit shell command.
    """
    def _builder(session) -> List[str]:
        tool = which_or_require(cmd[0], f"install '{cmd[0]}' via system package manager")
        # NOTE: for step 1 we rely on explicit commands (no wildcards).
        try:
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise ResourceRequired(f"command_failed:{cmd}", how_to_get=e.stdout.decode("utf-8", "ignore"))
        # store stdout as artifact
        out_digest = session.cas.put_bytes(proc.stdout, meta={"builder": "shell", "cmd": cmd})
        return [out_digest]
    return _builder

def text_render_builder(text: str, params: dict | None = None) -> callable:
    def _builder(session) -> List[str]:
        s = text.format(**(params or {}))
        digest = session.cas.put_bytes(s.encode("utf-8"), meta={"builder": "text.render"})
        return [digest]
    return _builder
