# engine/integrations_registry.py
from __future__ import annotations
from typing import Dict, Any, Callable
from adapters.android_builder import build_android_gradle
from adapters.ios_builder import build_ios_xcode
from adapters.unity_cli import build_unity
from adapters.cuda_runner import run_cuda_kernel
from adapters.k8s_plugin import submit_k8s_job
from adapters.contracts.base import ResourceRequired, ProcessFailed, BuildResult


REGISTRY: Dict[str, Callable[..., Any]] = {
    "android.build": build_android_gradle,
    "ios.build": build_ios_xcode,
    "unity.build": build_unity,
    "cuda.run": run_cuda_kernel,
    "k8s.job": submit_k8s_job,
}