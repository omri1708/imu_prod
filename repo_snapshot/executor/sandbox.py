# -*- coding: utf-8 -*-
from __future__ import annotations
import os, shlex, asyncio, signal, time, shutil, tempfile, resource, sys, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

from assurance.errors import ResourceRequired, ValidationFailed
from .policy import Policy

@dataclass
class Limits:
    cpu_seconds: int = 30
    mem_bytes: int = 256 * 1024 * 1024
    wall_seconds: int = 45
    open_files: int = 256
    no_net: bool = True  # block network (unshare or docker/bwrap)
    nice: int = 5

def _set_rlimits(lim: Limits):
    resource.setrlimit(resource.RLIMIT_CPU, (lim.cpu_seconds, lim.cpu_seconds))
    resource.setrlimit(resource.RLIMIT_NOFILE, (lim.open_files, lim.open_files))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (lim.mem_bytes, lim.mem_bytes))
    except Exception:
        pass
    try:
        os.nice(lim.nice)
    except Exception:
        pass

def _safe_env(base: Dict[str,str], allow_keys: List[str]) -> Dict[str,str]:
    env = {}
    for k in allow_keys:
        if k in base:
            env[k] = base[k]
    env["PYTHONUNBUFFERED"]="1"
    return env

def _which_or_req(tool: str, hint: str) -> str:
    p = shutil.which(tool)
    if not p:
        raise ResourceRequired(f"tool:{tool}", hint)
    return p

class SandboxExecutor:
    """
    Universal Executor with Policy:
      - tool signature/args policy
      - filesystem sandbox (bubblewrap if strict_fs), readonly system folders
      - network off by default (unshare/bwrap/docker)
      - rlimits + throttling
    """
    def __init__(self, policy_path: str = "./executor/policy.yaml", workdir_root: str = "./run_sandboxes"):
        self.root = Path(workdir_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = Policy.load(policy_path)
        self._semaphore = asyncio.Semaphore(8)

    async def run(self, argv: List[str], inputs: Dict[str, bytes] | None = None,
                  allow_write: List[str] | None = None, limits: Limits | None = None,
                  backend: str = "native", docker_image: str | None = None,
                  env: Dict[str,str] | None = None) -> Tuple[int, bytes]:
        async with self._semaphore:
            return await self._run_inner(argv, inputs or {}, allow_write or [], limits or self._limits_from_policy(),
                                         backend, docker_image, env or {})

    def _limits_from_policy(self) -> Limits:
        return Limits(
            cpu_seconds=self.policy.cpu_seconds,
            mem_bytes=self.policy.mem_bytes,
            wall_seconds=self.policy.wall_seconds,
            open_files=self.policy.open_files,
            no_net=self.policy.no_net_default
        )

    async def _run_inner(self, argv: List[str], inputs: Dict[str, bytes],
                         allow_write: List[str], limits: Limits,
                         backend: str, docker_image: str | None, env: Dict[str,str]) -> Tuple[int, bytes]:
        if not argv: raise ValidationFailed("empty argv")
        exe = shutil.which(argv[0])
        if not exe:
            raise ResourceRequired(f"tool:{argv[0]}", f"install '{argv[0]}'")
        # tool policy guard
        rule = self.policy.tool_guard(exe, argv)

        # prepare sandbox FS
        sb = Path(tempfile.mkdtemp(prefix="sbx_", dir=str(self.root)))
        (sb / "in").mkdir(); (sb / "out").mkdir(); (sb / "tmp").mkdir()
        for name, data in inputs.items():
            p = (sb / "in" / name).resolve()
            if sb not in p.parents: raise ValidationFailed("input path escapes sandbox")
            with open(p, "wb") as f: f.write(data)
        for a in allow_write:
            ap = (sb / a).resolve(); ap.parent.mkdir(parents=True, exist_ok=True)

        # backend selection
        if backend == "docker":
            docker = _which_or_req("docker", "install Docker")
            img = docker_image or "python:3.11-slim"
            full = [docker, "run", "--rm", "--network=none",
                    "-v", f"{sb}:/work:rw", "-w", "/work", img] + argv
            return await self._exec(full, cwd=str(sb), limits=limits, env=_safe_env(env, self.policy.allow_env), wall=limits.wall_seconds)
        else:
            # prefer bubblewrap when strict_fs
            if self.policy.strict_fs:
                bwrap = shutil.which("bwrap")
                if not bwrap:
                    raise ResourceRequired("tool:bwrap", "install bubblewrap or set strict_fs=false")
                # mount readonly system paths, bind sandbox to /work
                ro_dirs = []
                for d in ("/usr","/bin","/lib","/lib64"):
                    if os.path.isdir(d): ro_dirs += ["--ro-bind", d, d]
                net_opt = ["--unshare-net"] if limits.no_net else []
                full = [bwrap, "--die-with-parent", "--new-session", *ro_dirs,
                        "--bind", str(sb), "/work", "--chdir", "/work",
                        "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", *net_opt, *argv]
                return await self._exec(full, cwd=str(sb), limits=limits, env=_safe_env(env, self.policy.allow_env), wall=limits.wall_seconds)
            else:
                # native + unshare if possible
                full = argv
                if limits.no_net and sys.platform.startswith("linux"):
                    unshare = shutil.which("unshare")
                    if unshare:
                        full = [unshare, "-n", "--"] + argv
                    else:
                        raise ResourceRequired("tool:unshare", "install 'unshare' or set backend=docker")
                return await self._exec(full, cwd=str(sb), limits=limits, env=_safe_env(env, self.policy.allow_env), wall=limits.wall_seconds)

    async def _exec(self, full_cmd: List[str], cwd: str, limits: Limits, env: Dict[str,str], wall: int) -> Tuple[int, bytes]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *full_cmd, cwd=cwd, env=env,
                preexec_fn=(lambda: _set_rlimits(limits)),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        except FileNotFoundError:
            raise ResourceRequired(f"cmd_not_found:{full_cmd[0]}", "ensure tool is installed")
        try:
            out = await asyncio.wait_for(proc.communicate(), timeout=wall)
        except asyncio.TimeoutError:
            proc.kill(); await proc.wait()
            return 124, b"timeout"
        return proc.returncode, (out[0] or b"")
