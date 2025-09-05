# engine/register_adapters.py
from __future__ import annotations
from adapters.android_build import build_gradle
from adapters.ios_build import build_xcode
from adapters.unity_cli import build_unity_project
from adapters.cuda_jobs import compile_cuda_kernel
from adapters.k8s_deploy import apply_manifests

ADAPTERS = {
    "android.build_gradle": build_gradle,
    "ios.build_xcode": build_xcode,
    "unity.build_project": build_unity_project,
    "cuda.compile_kernel": compile_cuda_kernel,
    "k8s.apply_manifests": apply_manifests,
}

def resolve_adapter(name: str):
    if name not in ADAPTERS:
        raise KeyError(f"unknown_adapter: {name}")
    return ADAPTERS[name]