# tests/test_adapters_smoke.py
import os, pytest
from engine.adapters import register_all
from engine.registry import get
from contracts.base import ResourceRequired

register_all()

def test_android_smoke(tmp_path):
    # requires a gradle Android project; expect ResourceRequired when missing
    fn = get("android.build")
    out = tmp_path/"app.apk"
    with pytest.raises(ResourceRequired):
        fn(str(out), project_dir="examples/android_app", variant="debug")

def test_ios_smoke(tmp_path):
    fn = get("ios.build")
    with pytest.raises(ResourceRequired):
        fn(str(tmp_path/"App.xcarchive"), project_path="examples/ios/App.xcodeproj", scheme="App")

def test_unity_smoke(tmp_path):
    fn = get("unity.build")
    with pytest.raises(ResourceRequired):
        fn(project_dir="examples/unity")

def test_cuda_cpu_ok(tmp_path):
    fn = get("cuda.vadd")
    res = fn(n=10000, use_gpu=False)
    assert res.ok
    assert os.path.exists(res.artifact_path)

def test_k8s_apply_list(tmp_path):
    apply = get("k8s.apply"); pods = get("k8s.pods")
    yaml_text = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: imu-smoke
data:
  hello: world
"""
    try:
        res = apply(yaml_text)
        assert res.ok
        res2 = pods()
        assert res2.ok
    except ResourceRequired:
        # kubectl not present in CI – acceptable
        pass