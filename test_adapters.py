import pytest, os
from engine.errors import ResourceRequired
from adapters.android.build import build_apk
from adapters.ios.build import build_xcarchive
from adapters.unity.cli import unity_batch
from adapters.cuda.runner import run_cuda_job
from adapters.k8s.deploy import canary_and_rollout

def test_android_env_missing_raises():
    with pytest.raises(ResourceRequired):
        build_apk("/tmp/unknown_android_project")

def test_ios_env_missing_raises():
    with pytest.raises(ResourceRequired):
        build_xcarchive("Demo.xcodeproj", "Demo", "/tmp/out")

def test_unity_env_missing_raises():
    with pytest.raises(ResourceRequired):
        unity_batch("/tmp/unityproj", "Builder.Build")

def test_cuda_env_missing_raises():
    with pytest.raises(ResourceRequired):
        run_cuda_job("jobs/cuda_task.py")

def test_k8s_env_missing_raises():
    with pytest.raises(ResourceRequired):
        canary_and_rollout("deploy.yaml", "canary.yaml")