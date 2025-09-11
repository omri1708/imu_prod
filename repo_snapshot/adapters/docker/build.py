# adapters/docker/build.py
# -*- coding: utf-8 -*-
import shutil, subprocess
from ..contracts import ResourceRequired

def docker_build(tag: str, context_dir: str = ".", file: str = "Dockerfile"):
    if not shutil.which("docker"):
        raise ResourceRequired("Docker CLI", "Install Docker and ensure 'docker' in PATH")
    subprocess.run(["docker", "build", "-t", tag, "-f", file, context_dir], check=True)
    return {"ok": True, "image": tag}