# tests/test_adapters_requirements.py
# -*- coding: utf-8 -*-
import pytest
from engine.adapter_registry import get_adapter
from common.exc import ResourceRequired

def _expect_requirements(kind: str):
    ad = get_adapter(kind)
    if ad.detect():
        assert isinstance(ad.requirements(), tuple)
    else:
        with pytest.raises(ResourceRequired):
            ad.build({"kind":kind}, user="anonymous", workspace="build/out", policy=None, ev_index=None)

def test_android_req():
    _expect_requirements("android")

def test_ios_req():
    _expect_requirements("ios")

def test_unity_req():
    _expect_requirements("unity")

def test_cuda_req():
    _expect_requirements("cuda")

def test_k8s_req():
    _expect_requirements("k8s")