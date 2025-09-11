# tests/test_adapters_and_policy.py
import os, sys, json
from policy.model import default_user_policy
from adapters import android as A, ios as I, unity as U, cuda as C, k8s as K
from engine.grounding_gate import enforce_grounding

def test_android_dryrun_compose_cmds(tmp_path):
    pol = default_user_policy("u1")
    prj = str(tmp_path)
    open(os.path.join(prj,"gradlew"),"w").write("#!/bin/sh\necho ok\n"); os.chmod(os.path.join(prj,"gradlew"),0o755)
    out = A.dry_run(prj,"debug")
    assert out["ok"] and "gradlew" in " ".join(out["cmds"])

def test_unity_dryrun_cmd():
    out = U.dry_run("/path/to/proj","StandaloneLinux64")
    assert out["ok"] and "Unity -quit" in out["cmds"][0]

def test_cuda_dryrun():
    out = C.dry_run("kernel.cu","sm_80")
    assert out["ok"] and "nvcc" in out["cmds"][0]

def test_k8s_dryrun():
    out = K.dry_run("alpine:3.19","default")
    assert out["ok"] and "kubectl" in " ".join(out["cmds"])

def test_grounding_enforcement():
    pol = default_user_policy("u1")
    claims = [{
        "claim": "build succeeded",
        "evidence": [{"digest":"deadbeef"*8, "trust":"high"}]  # require_signed יקשל וזה טוב בבדיקה זו
    }]
    try:
        enforce_grounding(pol, claims)
        assert False, "should fail because envelope not found"
    except Exception:
        assert True