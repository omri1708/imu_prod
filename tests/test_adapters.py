# tests/test_adapters.py
import os
import types
from pathlib import Path
import builtins
import subprocess
import shutil
import pytest

from contracts.base import ResourceRequired, ContractError, Artifact
from provenance.store import ProvenanceStore

# ---- Helpers to monkeypatch external tool presence ----

class DummyProc:
    def __init__(self, rc=0, out="", err=""):
        self.returncode = rc
        self.stdout = out
        self.stderr = err

def fake_run_ok_success(cmd, cwd=None, env=None):
    return DummyProc(0, out="ok", err="")

def fake_run_ok_fail(cmd, cwd=None, env=None):
    return DummyProc(1, out="no", err="error")

def patch_which(monkeypatch, mapping):
    def _which(name):
        return mapping.get(name)
    monkeypatch.setattr(shutil, "which", _which)

def patch_run(monkeypatch, ok=True):
    def _run(cmd, cwd=None, env=None, capture_output=True, text=True):
        return DummyProc(0 if ok else 1, out="stdout", err="stderr")
    monkeypatch.setattr(subprocess, "run", _run)

# ---- Android ----
def test_android_requires_sdk(monkeypatch, tmp_path):
    from adapters import android_build as A
    # Simulate java present but no ANDROID_SDK_ROOT
    patch_which(monkeypatch, {"java": "/usr/bin/java"})
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    with pytest.raises(ResourceRequired):
        A.build_gradle(project_dir=str(tmp_path))

def test_android_gradle_wrapper(monkeypatch, tmp_path):
    from adapters import android_build as A
    # Project structure
    (tmp_path / "app" / "build" / "outputs" / "apk").mkdir(parents=True)
    apk = tmp_path / "app" / "build" / "outputs" / "apk" / "app-release.apk"
    apk.write_bytes(b"FAKE")
    # gradlew present
    (tmp_path / "gradlew").write_text("#!/bin/sh\necho build\n")
    os.chmod(tmp_path / "gradlew", 0o755)

    patch_which(monkeypatch, {"java": "/usr/bin/java"})
    monkeypatch.setenv("ANDROID_SDK_ROOT", "/opt/android-sdk")
    patch_run(monkeypatch, ok=True)

    store = ProvenanceStore(root=str(tmp_path / ".prov"))
    art = A.build_gradle(str(tmp_path), store=store)
    assert isinstance(art, Artifact)
    assert art.kind == "apk"
    assert art.provenance_sha256

# ---- iOS ----
@pytest.mark.skipif(os.name != "posix", reason="requires mac host")
def test_ios_missing_xcode(monkeypatch, tmp_path):
    from adapters import ios_build as I
    patch_which(monkeypatch, {})  # no xcodebuild
    with pytest.raises(ResourceRequired):
        I.build_xcode(str(tmp_path), scheme="App")

# ---- Unity ----
def test_unity_requires_path(monkeypatch, tmp_path):
    from adapters import unity_cli as U
    monkeypatch.delenv("UNITY_PATH", raising=False)
    with pytest.raises(ResourceRequired):
        U.build_unity_project(str(tmp_path))

# ---- CUDA ----
def test_cuda_requires_nv(monkeypatch, tmp_path):
    from adapters import cuda_jobs as C
    patch_which(monkeypatch, {})  # no nvcc/nvidia-smi
    with pytest.raises(ResourceRequired):
        C.compile_cuda_kernel(str(tmp_path / "k.cu"))

# ---- K8s ----
def test_k8s_apply_missing_kubectl(monkeypatch, tmp_path):
    from adapters import k8s_deploy as K
    (tmp_path / "d.yaml").write_text("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: d\n")
    patch_which(monkeypatch, {})  # kubectl missing
    with pytest.raises(ResourceRequired):
        K.apply_manifests([str(tmp_path / "d.yaml")])