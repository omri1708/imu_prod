# tests/test_contracts_and_adapters.py
# -*- coding: utf-8 -*-
import os, json, time, pytest
from engine.contracts_gate import enforce_respond_contract, ContractError, EvidenceMissing
from governance.user_policy import get_user_policy
from evidence import cas
from engine.adapter_registry import get_adapter
from common.errors import ResourceRequired

def test_contracts_enforce_ok():
    policy, ev_index = get_user_policy("alice")
    ev = {"doc":"ok"}
    sha = cas.put_json(ev)
    claims = [{"id":"x.temp","value":25.0,"schema":{"type":"float","min":-50,"max":150,"unit":"C"},
               "evidence":[{"sha256":sha,"fetched_at":time.time(),"ttl_sec":1e9,"trust":0.7}]}]
    enforce_respond_contract(stage="test", claims=claims, evidence=[{"sha256":sha}], policy=policy, ev_index=ev_index)

def test_contracts_missing_ev():
    policy, ev_index = get_user_policy("alice")
    claims = [{"id":"x.speed","value":120,"schema":{"type":"int","min":0,"max":300},"evidence":[]}]
    with pytest.raises(ContractError):
        enforce_respond_contract(stage="test", claims=claims, evidence=[], policy=policy, ev_index=ev_index)

def test_k8s_adapter_resource_required(tmp_path):
    policy, ev_index = get_user_policy("alice")
    # kubectl או kubeconfig לרוב לא יהיו ב-CI: מצפה ל-ResourceRequired
    from engine.adapter_registry import get_adapter
    ad = get_adapter("k8s")
    with pytest.raises(ResourceRequired) as ei:
        ad.build({"kind":"k8s","manifest":"apiVersion: v1\nkind: Namespace\nmetadata:\n  name: imu"}, "alice", str(tmp_path), policy, ev_index)
    assert "kubectl" in str(ei.value)

def test_cuda_adapter_resource_required(tmp_path):
    policy, ev_index = get_user_policy("alice")
    ad = get_adapter("cuda")
    try:
        ad.build({"kind":"cuda","source":"__global__ void k(){}"}, "alice", str(tmp_path), policy, ev_index)
        # אם מותקן nvcc – יעבור; אחרת ResourceRequired זה גם תקין
    except ResourceRequired:
        pass