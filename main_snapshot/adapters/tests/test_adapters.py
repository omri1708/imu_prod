# adapters/tests/test_adapters.py
# -*- coding: utf-8 -*-
import os, shutil, pytest
from adapters.contracts import ResourceRequired
from adapters.k8s.deploy import deploy
from adapters.gpu.cuda_runner import compile_and_run_cuda

def _skip_if_missing(tool):
    if shutil.which(tool) is None:
        pytest.skip(f"missing {tool}")

def test_k8s_deploy_manifest_only(tmp_path, monkeypatch):
    # אם kubectl חסר — נוודא שמקבלים ResourceRequired
    if shutil.which("kubectl") is None:
        with pytest.raises(ResourceRequired):
            deploy("demo", "nginx:alpine")
    else:
        r = deploy("demo", "nginx:alpine")
        assert "manifest" in r and os.path.exists(r["manifest"])

def test_cuda_compile_or_require():
    if shutil.which("nvcc") is None:
        with pytest.raises(ResourceRequired):
            compile_and_run_cuda()
    else:
        out = compile_and_run_cuda()
        assert "y[0]=" in out["output"]