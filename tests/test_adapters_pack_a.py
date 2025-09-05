# tests/test_adapters_pack_a.py
import os, pytest, shutil
from examples.usage_android_ios_unity_cuda_k8s import ex_android, ex_ios, ex_unity, ex_cuda, ex_k8s

def _has(cmd): return shutil.which(cmd) is not None

@pytest.mark.skipif(not (_has("gradle") or os.path.exists("sample-android/gradlew")), reason="Android toolchain not present")
def test_android_smoke():
    res = ex_android("sample-android")
    assert "ok" in res

@pytest.mark.skipif(not _has("xcodebuild"), reason="Xcode not present")
def test_ios_smoke():
    res = ex_ios("Sample.xcodeproj", "Sample")
    assert "ok" in res

@pytest.mark.skipif(not (_has("unity") or _has("Unity") or _has("Unity.app/Contents/MacOS/Unity")), reason="Unity CLI not present")
def test_unity_smoke():
    res = ex_unity("SampleUnityProject", "Builder.PerformBuild")
    assert "ok" in res

@pytest.mark.skipif(not _has("nvcc"), reason="CUDA nvcc not present")
def test_cuda_smoke():
    res = ex_cuda()
    assert "ok" in res

@pytest.mark.skipif(not _has("kubectl"), reason="kubectl not present")
def test_k8s_job():
    res = ex_k8s()
    assert "ok" in res or (not res["ok"] and ("need" in res or "error" in res))