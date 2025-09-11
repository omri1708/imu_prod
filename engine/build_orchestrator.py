# PATH: engine/build_orchestrator.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
import hashlib
import json
import subprocess
import os
import tempfile
from asyncio.subprocess import PIPE, STDOUT
from pathlib import Path
from engine.artifacts.registry import register as register_artifacts
from engine.self_heal.auto_pr import run as auto_pr_run

"""
BuildOrchestrator (hybrid)
- Primary path: run inside a restricted SandboxExecutor (no_net) if available
- Fallback path: run in an ephemeral tmpdir with asyncio.subprocess
- Always returns a rich result dict; never raises on normal build errors
- Adds SpecLock digest + compact manifest (like the tmpdir version)
"""


# Sandbox is optional; if missing we transparently use the tmpdir runner
try:
    from executor.sandbox import SandboxExecutor, Limits  # type: ignore
except Exception:  # pragma: no cover
    SandboxExecutor = None  # type: ignore
    Limits = None  # type: ignore

BytesLike = Union[bytes, str]


def _persist_files(files: Dict[str, Any], root: str) -> List[str]:
    root_p = Path(root)
    written: List[str] = []
    for rel, data in files.items():
        p = root_p / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            data = data.encode("utf-8")
        p.write_bytes(data)
        written.append(str(p))
    return written
def _decode(b: Optional[bytes]) -> str:
    return (b or b"").decode("utf-8", "ignore")


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _ensure_bytes_map(files: Dict[str, BytesLike]) -> Dict[str, bytes]:
    out: Dict[str, bytes] = {}
    for path, data in files.items():
        if isinstance(data, str):
            out[path] = data.encode("utf-8")
        else:
            out[path] = data
    return out


def _spec_digest_from_inputs(inputs: Dict[str, bytes]) -> str:
    # deterministic over sorted filenames + text content (utf-8, ignore)
    payload = {k: inputs[k].decode("utf-8", "ignore") for k in sorted(inputs)}
    return _sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _collect_py_targets(inputs: Dict[str, bytes]) -> Tuple[List[str], List[str]]:
    py_targets = [p for p in inputs.keys() if p.endswith(".py")]
    test_files = [
        p for p in py_targets
        if p.startswith("test") or p.endswith("_test.py") or "/test" in p or "/tests/" in p or "\\test" in p
    ]
    return py_targets, test_files


class BuildOrchestrator:
    """
    Build Python modules from {filename: content} with:
      1) py_compile for all .py files
      2) pytest -q when tests exist and pytest is installed

    Primary mode: sandbox (no_net). Fallback: tmpdir subprocess.
    Never raises on build failures; returns a structured dict.
    """

    def __init__(
        self,
        policy_path: str = "./executor/policy.yaml",
        workroot: str = "./program_sbx",
        *,
        prefer_sandbox: bool = True,
        default_limits: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.prefer_sandbox = bool(prefer_sandbox)
        self.exec = None
        if SandboxExecutor is not None and self.prefer_sandbox:
            try:
                self.exec = SandboxExecutor(policy_path, workroot)  # type: ignore[call-arg]
            except Exception:
                self.exec = None
        self.default_limits = default_limits or {"no_net": True}

    async def build_python_module(
        self,
        files: Dict[str, BytesLike],
        name: str = "glue",
        *,
        lint: bool = True,
        run_tests: Optional[bool] = None,
        run_e2e: bool = False,
        e2e_dir: str = "tests",
        e2e_kexpr: str = "e2e or end2end or integration",
        use_tmpdir: Optional[bool] = None,
        pytest_args: Optional[List[str]] = None,
        e2e_args: Optional[List[str]] = None,
        timeout_s: Optional[float] = 60.0,
        cpu_seconds: Optional[int] = None,
        mem_mb: Optional[int] = None,
        workdir: Optional[str] = None,
        persist_dir: Optional[str] = None,
        
    ) -> Dict[str, Any]:
        inputs = _ensure_bytes_map(files)
        py_targets, test_files = _collect_py_targets(inputs)
        if not py_targets:
            return {"ok": False, "error": "no_python_files", "compile_rc": None, "compile_out": "", "module": name}

        spec_digest = _spec_digest_from_inputs(inputs)
        run_tests_env = os.getenv("IMU_RUN_TESTS", "0").strip().lower() in ("1","true","yes","on")
        run_tests_flag = run_tests if run_tests is not None else run_tests_env
        fail_on_tests = os.getenv("IMU_FAIL_ON_TESTS", "0").strip().lower() in ("1","true","yes","on")

        # choose runner
        force_tmp = (use_tmpdir is True) or (self.exec is None and self.prefer_sandbox)
        if not force_tmp and self.exec is not None:
            result = await self._build_with_sandbox(
                inputs, py_targets, test_files,
                lint, run_tests_flag, run_e2e, pytest_args, e2e_args,
                e2e_dir, e2e_kexpr, timeout_s, cpu_seconds, mem_mb
            )
        else:
            result = await self._build_with_tmpdir(
                inputs, py_targets, test_files,
                lint, run_tests_flag, run_e2e, pytest_args, e2e_args,
                e2e_dir, e2e_kexpr, timeout_s, cpu_seconds, mem_mb,
                base_dir=workdir  # ← אם נתת workdir, נשתמש בו במקום tmpdir
            )

        # attach digest + manifest
        manifest = {
            "name": name,
            "spec_digest": spec_digest,
            "files": sorted(py_targets),
            "compile_rc": result.get("compile_rc"),
            "test_rc": result.get("test_rc"),
        }
        result.update({
            "module": name,
            "spec_digest": spec_digest,
            "manifest": manifest,
            "files_built": sorted(py_targets),
        })
        # allow API to persist the generated sources:
        result["inputs"] = inputs
        
        # Persist קבצים אם ביקשת
        if (persist_dir or workdir) and isinstance(inputs, dict):
            try:
                target = persist_dir or workdir
                written = _persist_files(inputs, target)
                result["persist_dir"] = target
                result["files_written"] = written
            except Exception as e:
                result["persist_error"] = f"{e}"
        try:
            build_dg = register_artifacts(f"build-{name}", inputs)   # inputs = mapping {path:bytes|str}
            result["build_artifact_digest"] = build_dg
        except Exception:
            pass

        if not result.get("ok"):
            title = f"Fix build failure: {result.get('name','module')}"
            body  = "Build failed.\n\n```\n" + (result.get("log","")[:4000]) + "\n```"
            try:
                pr = auto_pr_run(repo_dir=".", title=title, body=body)
                result["auto_pr"] = pr
            except Exception as e:
                result["auto_pr_error"] = str(e)
        # make original inputs available to callers (for persist)
        

        return result

    # ---------------------------- sandbox runner ----------------------------
    async def _build_with_sandbox(
        self,
        inputs: Dict[str, bytes],
        py_targets: List[str],
        test_files: List[str],
        lint: bool,
        run_tests: bool,
        run_e2e: bool,
        pytest_args: Optional[List[str]],
        e2e_args: Optional[List[str]],
        e2e_dir: str,
        e2e_kexpr: str,
        timeout_s: Optional[float],
        cpu_seconds: Optional[int],
        mem_mb: Optional[int],
    ) -> Dict[str, Any]:
        assert self.exec is not None, "Sandbox executor is not initialized"

        # helper: build Limits (best-effort) and run with optional timeout
        try:
            kw = dict(self.default_limits)
            if cpu_seconds is not None:
                kw["cpu_seconds"] = int(cpu_seconds)
            if mem_mb is not None:
                kw["mem_bytes"] = int(mem_mb) * 1024 * 1024
            if timeout_s is not None:
                kw["wall_seconds"] = int(timeout_s)
            limits = Limits(**kw) if Limits is not None else None  # type: ignore[arg-type]
        except Exception:
            limits = None
        async def _sx(cmd, **extra):
            kwargs = dict(inputs=inputs, allow_write=[".", "__pycache__", "tests", "test"], limits=limits)
            kwargs.update(extra)
            try:
                return await self.exec.run(cmd, timeout_s=timeout_s, **kwargs)  # type: ignore[call-arg]
            except TypeError:
                return await self.exec.run(cmd, **kwargs)

        # phase 0: lint (ruff/flake8) if requested
        lint_rc, lint_out = 0, "skipped"
        if lint:
            # detect ruff/flake8
            rc_chk, _ = await _sx(["python","-c","import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('ruff') else 43)"])
            if rc_chk == 0:
                lint_rc, out = await _sx(["python","-m","ruff","-q","." ])
                lint_out = _decode(out)
            else:
                rc_chk2, _ = await _sx(["python","-c","import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('flake8') else 44)"])
                if rc_chk2 == 0:
                    lint_rc, out = await _sx(["python","-m","flake8","."])
                    lint_out = _decode(out)
                else:
                    lint_rc, lint_out = 0, "linters not installed; skipped"

        # 1) py_compile all targets
        rc1, out1 = await _sx(["python", "-m", "py_compile", *py_targets])
        out1s = _decode(out1)

        # 2) pytest (optional unit tests)
        rc2, out2s = 0, ""
        if run_tests and test_files:
            rc_chk, _ = await _sx(["python","-c","import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('pytest') else 42)"])
            if rc_chk == 0:               
                pytest_target = "."
                if any(p.startswith("services/api/tests") for p in test_files):
                    pytest_target = "services/api/tests"
                elif any(p.startswith("tests") for p in test_files):
                    pytest_target = "tests"
                cmd = ["python","-m","pytest","-q", pytest_target] + (pytest_args or [])
                rc2, out2 = await _sx(cmd)
                out2s = _decode(out2)
            else:
                rc2, out2s = 0, "pytest not installed; skipped"
        elif not test_files:
            rc2, out2s = 0, "no tests detected; skipped"
        
        # 3) e2e (optional)
        rc3, out3s = 0, ""
        if run_e2e:
            rc_chk, _ = await _sx(["python","-c","import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('pytest') else 42)"])
            if rc_chk == 0:
                cmd = ["python","-m","pytest","-q", e2e_dir, "-k", e2e_kexpr] + (e2e_args or [])
                rc3, out3 = await _sx(cmd)
                out3s = _decode(out3)
            else:
                rc3, out3s = 0, "pytest not installed; skipped"
        
        fail_on_tests = os.getenv("IMU_FAIL_ON_TESTS","0").strip().lower() in ("1","true","yes","on")
        ok = (lint_rc == 0 and rc1 == 0 and (not fail_on_tests or rc2 == 0) and (not run_e2e or rc3 == 0))

        return {
            "ok": ok,
            "lint_rc": lint_rc, "lint_out": lint_out,
            "compile_rc": rc1,
            "compile_out": out1s,
            "test_rc": rc2,
            "test_out": out2s,
            "e2e_rc": rc3,
            "e2e_out": out3s,
        }


    # ---------------------------- tmpdir runner -----------------------------
    async def _build_with_tmpdir(
        self,
        inputs: Dict[str, bytes],
        py_targets: List[str],
        test_files: List[str],
        lint: bool,
        run_tests: bool,
        run_e2e: bool,
        pytest_args: Optional[List[str]],
        e2e_args: Optional[List[str]],
        e2e_dir: str,
        e2e_kexpr: str,
        timeout_s: Optional[float],
        cpu_seconds: Optional[int],
        mem_mb: Optional[int],
        base_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        compile_rc, compile_out = 1, ""
        test_rc, test_out = 0, ""
        try:
            # אם קיבלת base_dir – נעבוד בו; אחרת tmpdir זמני
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)
                td = base_dir
                def _cleanup(): pass
            else:
                tmp = tempfile.TemporaryDirectory()
                td = tmp.name
                def _cleanup(): tmp.cleanup()
            try:
                # write all files to tmpdir (preserve nested paths)
                for rel, data in inputs.items():
                    abspath = os.path.join(td, rel)
                    os.makedirs(os.path.dirname(abspath) or td, exist_ok=True)
                    with open(abspath, "wb") as f:
                        f.write(data)

                # preexec_fn for POSIX limits
                preexec = None
                if os.name == "posix" and (cpu_seconds or mem_mb):
                    import resource  # type: ignore
                    def _pre():
                        if cpu_seconds:
                            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
                        if mem_mb:
                            bytes_ = int(mem_mb) * 1024 * 1024
                            for r in (resource.RLIMIT_AS, getattr(resource, "RLIMIT_DATA", None)):
                                if r is not None:
                                    resource.setrlimit(r, (bytes_, bytes_))
                    preexec = _pre

                # phase 0: lint
                lint_rc, lint_out = 0, "skipped"
                if lint:
                    # prefer ruff, fallback to flake8
                    chk = await asyncio.create_subprocess_exec(
                        "python","-c","import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('ruff') else 43)",
                        cwd=td, stdout=PIPE, stderr=STDOUT, preexec_fn=preexec
                    )
                    _o,_ = await chk.communicate()
                    if chk.returncode == 0:
                        p_l = await asyncio.create_subprocess_exec(
                            "python","-m","ruff","-q",".", cwd=td, stdout=PIPE, stderr=STDOUT, preexec_fn=preexec
                        )
                        out,_ = await asyncio.wait_for(p_l.communicate(), timeout=timeout_s)
                        lint_rc, lint_out = p_l.returncode, _decode(out)
                    else:
                        chk2 = await asyncio.create_subprocess_exec(
                            "python","-c","import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('flake8') else 44)",
                            cwd=td, stdout=PIPE, stderr=STDOUT, preexec_fn=preexec
                        )
                        _o,_ = await chk2.communicate()
                        if chk2.returncode == 0:
                            p_l = await asyncio.create_subprocess_exec(
                                "python","-m","flake8"," .", cwd=td, stdout=PIPE, stderr=STDOUT, preexec_fn=preexec
                            )
                            out,_ = await asyncio.wait_for(p_l.communicate(), timeout=timeout_s)
                            lint_rc, lint_out = p_l.returncode, _decode(out)
                        else:
                            lint_rc, lint_out = 0, "linters not installed; skipped"

                # 1) py_compile
                p = await asyncio.create_subprocess_exec(
                    "python", "-m", "py_compile", *py_targets,
                    cwd=td, stdout=PIPE, stderr=STDOUT, preexec_fn=preexec,
                )
                out, _ = await asyncio.wait_for(p.communicate(), timeout=timeout_s)
                compile_rc, compile_out = p.returncode, _decode(out)

                # 2) pytest (optional)
                if run_tests and test_files:
                    # יעד חכם לטסטים
                    if os.path.isdir(os.path.join(td, "services", "api", "tests")):
                        pytest_target = os.path.join("services", "api", "tests")
                    elif os.path.isdir(os.path.join(td, "tests")):
                        pytest_target = "tests"
                    else:
                        dirs = sorted(set(os.path.dirname(p) or "." for p in test_files))
                        pytest_target = dirs[0] if len(dirs) == 1 else "."

                    chk = await asyncio.create_subprocess_exec(
                        "python", "-c",
                        "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('pytest') else 42)",
                        cwd=td, stdout=PIPE, stderr=STDOUT, preexec_fn=preexec,
                    )
                    _o, _ = await chk.communicate()
                    if chk.returncode == 0:
                        p2 = await asyncio.create_subprocess_exec(
                            "python", "-m", "pytest", "-q", pytest_target,
                            cwd=td, stdout=PIPE, stderr=STDOUT, preexec_fn=preexec
                        )
                        out2, _ = await asyncio.wait_for(p2.communicate(), timeout=timeout_s)
                        test_rc, test_out = p2.returncode, _decode(out2)
                    else:
                        test_rc, test_out = 0, "pytest not installed; skipped"
                elif not test_files:
                    test_rc, test_out = 0, "no tests detected; skipped"
                # 3) e2e
                e2e_rc, e2e_out = 0, ""
                if run_e2e:
                    chk = await asyncio.create_subprocess_exec(
                        "python","-c","import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('pytest') else 42)",
                        cwd=td, stdout=PIPE, stderr=STDOUT, preexec_fn=preexec
                    )
                    _o,_ = await chk.communicate()
                    if chk.returncode == 0:
                        cmd = ["python","-m","pytest","-q","tests","-k","e2e or end2end or integration"] + (e2e_args or []) 
                        p3 = await asyncio.create_subprocess_exec(*cmd, cwd=td, stdout=PIPE, stderr=STDOUT, preexec_fn=preexec)
                        out3,_ = await asyncio.wait_for(p3.communicate(), timeout=timeout_s)
                        e2e_rc, e2e_out = p3.returncode, _decode(out3)
                    else:
                        e2e_rc, e2e_out = 0, "pytest not installed; skipped"
            finally:
                _cleanup()
        except Exception as e:
            # fall back to structured failure without raising
            return {
                "ok": False,
                "compile_rc": compile_rc,
                "compile_out": f"tmpdir_error: {e}",
                "test_rc": test_rc,
                "test_out": test_out,
            }

        fail_on_tests = os.getenv("IMU_FAIL_ON_TESTS","0").strip().lower() in ("1","true","yes","on")
        ok = (lint_rc == 0 and compile_rc == 0 and (not fail_on_tests or test_rc == 0) and (not run_e2e or e2e_rc == 0))
        return {
             "ok": ok,
            "lint_rc": lint_rc, "lint_out": lint_out,
             "compile_rc": compile_rc,
             "compile_out": compile_out,
             "test_rc": test_rc,
             "test_out": test_out,
            "e2e_rc": e2e_rc,
            "e2e_out": e2e_out,
         }