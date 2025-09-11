# tests/test_adapters_packA.py
# -*- coding: utf-8 -*-
import os, json, tempfile, pytest
from engine.adapter_registry import get_adapter
from common.exc import ResourceRequired

def test_k8s_adapter_dryrun_or_requirements():
    ad = get_adapter("k8s")
    ws = tempfile.mkdtemp(prefix="imu_k8s_")
    # manifest מינימלי (אם אין kubectl/cluster נקבל ResourceRequired)
    job = {"kind":"k8s","manifest":"apiVersion: v1\nkind: Namespace\nmetadata:\n  name: imu-demo\n"}
    try:
        res = ad.build(job, user="test", workspace=ws, policy=None, ev_index=None)
        assert res.artifacts and res.claims
    except ResourceRequired as rr:
        assert "kubectl" in rr.how_to or "kube" in rr.kind

def test_android_adapter_prereqs():
    ad = get_adapter("android")
    ws = tempfile.mkdtemp(prefix="imu_andr_")
    with pytest.raises(ResourceRequired):
        # יחזור ResourceRequired אם חסר gradle/javac
        ad.build({"kind":"android"}, user="t", workspace=ws, policy=None, ev_index=None)

def test_ios_adapter_prereqs():
    ad = get_adapter("ios")
    ws = tempfile.mkdtemp(prefix="imu_ios_")
    with pytest.raises(ResourceRequired):
        ad.build({"kind":"ios"}, user="t", workspace=ws, policy=None, ev_index=None)

def test_unity_adapter_prereqs():
    ad = get_adapter("unity")
    ws = tempfile.mkdtemp(prefix="imu_unity_")
    with pytest.raises(ResourceRequired):
        ad.build({"kind":"unity"}, user="t", workspace=ws, policy=None, ev_index=None)

def test_cuda_adapter_prereqs():
    ad = get_adapter("cuda")
    ws = tempfile.mkdtemp(prefix="imu_cuda_")
    with pytest.raises(ResourceRequired):
        ad.build({"kind":"cuda"}, user="t", workspace=ws, policy=None, ev_index=None)