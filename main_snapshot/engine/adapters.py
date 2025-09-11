# engine/adapters.py
from engine.registry import register
from adapters.android.build_android import build as android_build
from adapters.ios.build_ios import build as ios_build
from adapters.unity.build_unity import build as unity_build
from adapters.cuda.job_runner import run_vector_add as cuda_vadd
from adapters.k8s.deploy_plugin import apply_manifest as k8s_apply, get_pods as k8s_pods

def register_all():
    register("android.build", android_build)
    register("ios.build", ios_build)
    register("unity.build", unity_build)
    register("cuda.vadd", cuda_vadd)
    register("k8s.apply", k8s_apply)
    register("k8s.pods", k8s_pods)