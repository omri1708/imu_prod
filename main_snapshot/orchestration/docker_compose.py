# imu_repo/orchestration/docker_compose.py
from __future__ import annotations
import os, shutil, subprocess, json
from typing import Dict, Any, List

from orchestration.services import ServiceSpec


class ResourceRequired(Exception):
    def __init__(self, what: str, how: str):
        super().__init__(f"resource_required:{what}")
        self.what=what; self.how=how

def has_docker() -> bool:
    return shutil.which("docker") is not None

def write_compose(path: str, services: Dict[str, Dict[str, Any]]):
    """
    services: {
      "redis": {"image":"redis:7", "ports":["6379:6379"]},
      "api": {"image":"nginx:alpine","ports":["8080:80"]}
    }
    """
    lines = ["version: '3.9'","services:"]
    for name, spec in services.items():
        lines.append(f"  {name}:")
        if "image" in spec:
            lines.append(f"    image: {spec['image']}")
        if "command" in spec:
            lines.append(f"    command: {spec['command']}")
        if "env" in spec:
            lines.append("    environment:")
            for k,v in spec["env"].items():
                lines.append(f"      - {k}={v}")
        if "ports" in spec:
            lines.append("    ports:")
            for p in spec["ports"]:
                lines.append(f"      - \"{p}\"")
        if "volumes" in spec:
            lines.append("    volumes:")
            for v in spec["volumes"]:
                lines.append(f"      - {v}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w") as f: f.write("\n".join(lines)+"\n")


def up(compose_path: str):
    if not has_docker():
        raise ResourceRequired("docker", "Install Docker and enable daemon")
    subprocess.run(["docker","compose","-f",compose_path,"up","-d"], check=True)


def down(compose_path: str):
    if not has_docker():
        return
    subprocess.run(["docker","compose","-f",compose_path,"down"], check=True)


class ComposeWriter:
    def __init__(self, root: str = ".imu_state/compose"):
        self.root=root; os.makedirs(root, exist_ok=True)

    def write(self, services: List[ServiceSpec], file_name: str="docker-compose.yml") -> str:
        path = os.path.join(self.root, file_name)
        obj={"version":"3.9","services":{s.name: s.to_compose() for s in services}}
        # לכתוב YAML ידנית כדי לא להזדקק ל-PyYAML
        def y(d, indent=0):
            sp="  "*indent
            if isinstance(d, dict):
                out=[]
                for k,v in d.items():
                    if isinstance(v,(dict,list)):
                        out.append(f"{sp}{k}:")
                        out.append(y(v, indent+1))
                    else:
                        out.append(f"{sp}{k}: {json.dumps(v)}")
                return "\n".join(out)
            if isinstance(d, list):
                out=[]
                for it in d:
                    if isinstance(it,(dict,list)):
                        out.append(f"{sp}-")
                        out.append(y(it, indent+1))
                    else:
                        out.append(f"{sp}- {json.dumps(it)}")
                return "\n".join(out)
            return f"{sp}{json.dumps(d)}"
        with open(path,"w",encoding="utf-8") as f:
            f.write(y(obj)+"\n")
        return path

    def up(self, compose_path: str):
        docker = shutil.which("docker") or shutil.which("docker.exe")
        if not docker:
            raise ResourceRequired("docker", "Install Docker Engine & compose plugin, then run: docker compose -f "+compose_path+" up -d")
        # נריץ compose
        cmd=[docker, "compose", "-f", compose_path, "up", "-d"]
        subprocess.check_call(cmd)

    def down(self, compose_path: str):
        docker = shutil.which("docker") or shutil.which("docker.exe")
        if not docker:
            raise ResourceRequired("docker", "Install Docker Engine & compose plugin, then run: docker compose -f "+compose_path+" down")
        cmd=[docker, "compose", "-f", compose_path, "down"]
        subprocess.check_call(cmd)
