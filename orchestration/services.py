# imu_repo/orchestration/services.py
from __future__ import annotations
import subprocess, time, socket, urllib.request, urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

class OrchestrationError(Exception): ...

@dataclass
class ServiceSpec:
    def __init__(self, name: str, image: str, ports: Optional[List[str]]=None, env: Optional[Dict[str,str]]=None, depends_on: Optional[List[str]]=None, volumes: Optional[List[str]]=None, command: Optional[List[str]]=None):
        self.name=name; self.image=image
        self.ports=ports or []; self.env=env or {}; self.depends_on=depends_on or []
        self.volumes=volumes or []; self.command=command or []
    
    def to_compose(self) -> Dict[str,Any]:
        o={"image": self.image}
        if self.ports: o["ports"]=self.ports
        if self.env: o["environment"]=self.env
        if self.depends_on: o["depends_on"]=self.depends_on
        if self.volumes: o["volumes"]=self.volumes
        if self.command: o["command"]=self.command
        return o
    
    name: str
    command: List[str]
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    http_health: Optional[str] = None  # "http://127.0.0.1:8001/health"
    tcp_health: Optional[tuple[str,int]] = None  # ("127.0.0.1", 8001)
    start_timeout_s: float = 15.0
    stop_timeout_s: float = 5.0

@dataclass
class RunningService:
    spec: ServiceSpec
    proc: subprocess.Popen

class Orchestrator:
    def __init__(self):
        self.running: Dict[str, RunningService] = {}

    def _probe_http(self, url: str, timeout=1.0) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return 200 <= r.getcode() < 400
        except Exception:
            return False

    def _probe_tcp(self, host: str, port: int, timeout=1.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    def _wait_healthy(self, spec: ServiceSpec) -> bool:
        t0 = time.time()
        while time.time() - t0 < spec.start_timeout_s:
            ok_h = True
            if spec.http_health:
                ok_h = self._probe_http(spec.http_health)
            ok_t = True
            if spec.tcp_health:
                ok_t = self._probe_tcp(*spec.tcp_health)
            if ok_h and ok_t:
                return True
            time.sleep(0.2)
        return False

    def start(self, spec: ServiceSpec):
        if spec.name in self.running:
            raise OrchestrationError(f"already_running:{spec.name}")
        proc = subprocess.Popen(spec.command, cwd=spec.cwd, env=spec.env)
        rs = RunningService(spec=spec, proc=proc)
        self.running[spec.name] = rs
        if not self._wait_healthy(spec):
            self.stop(spec.name)
            raise OrchestrationError(f"healthcheck_failed:{spec.name}")

    def stop(self, name: str):
        rs = self.running.pop(name, None)
        if not rs:
            return
        rs.proc.terminate()
        try:
            rs.proc.wait(timeout=rs.spec.stop_timeout_s)
        except Exception:
            rs.proc.kill()
