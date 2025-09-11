# tests/test_adapters_b.py
# -*- coding: utf-8 -*-
import pytest
from adapters.contracts import ResourceRequired
from adapters.gpu.cuda_runner import compile_and_run_cuda

def test_cuda_optional_env():
    try:
        r = compile_and_run_cuda(devices=1)
    except ResourceRequired:
        pytest.skip("CUDA not installed")
    else:
        assert "OK" in r["results"][0]