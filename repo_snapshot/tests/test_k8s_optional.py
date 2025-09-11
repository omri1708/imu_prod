# tests/test_k8s_optional.py
# -*- coding: utf-8 -*-
import pytest
from adapters.k8s.deploy import apply_manifest, ResourceRequired

def test_k8s_apply_optional():
    m = {"apiVersion":"v1","kind":"Namespace","metadata":{"name":"imu-test-ns"}}
    try:
        r = apply_manifest(m)
    except ResourceRequired:
        pytest.skip("kubectl not installed or no context")
    else:
        assert r["ok"]