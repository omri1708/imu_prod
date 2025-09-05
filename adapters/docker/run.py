# adapters/docker/run.py
# -*- coding: utf-8 -*-
import shutil, subprocess
from ..contracts import ResourceRequired

def docker_run(tag: str, port_map: str = None, env: dict = None, detach: bool = True):
    if not shutil.which("docker"):
        raise ResourceRequired("Docker CLI", "Install Docker and ensure 'docker' in PATH")
    cmd = ["docker", "run"]
    if detach: cmd.append("-d")
    if port_map: cmd += ["-p", port_map]
    if env:
        for k,v in env.items(): cmd += ["-e", f"{k}={v}"]
    cmd.append(tag)
    subprocess.run(cmd, check=True)
    return {"ok": True}