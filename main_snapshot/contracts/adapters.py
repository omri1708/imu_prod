# contracts/adapters.py — חוזים מחייבים (בדיקות מוקדמות)
# -*- coding: utf-8 -*-
import shutil, os, subprocess
from typing import Tuple
from engine.errors import ResourceRequired

def _require(cmd: str, install_hint: str):
    if shutil.which(cmd) is None:
        raise ResourceRequired(cmd, install_hint)

def android_env():
    _require("javac", "Install JDK (e.g., Temurin). Ensure JAVA_HOME and PATH.")
    if not os.environ.get("ANDROID_HOME") and not os.environ.get("ANDROID_SDK_ROOT"):
        raise ResourceRequired("Android SDK", "Install Android SDK + cmdline-tools; set ANDROID_SDK_ROOT.")
    _require("gradle", "Install Gradle or use ./gradlew in project.")

def ios_env():
    _require("xcodebuild", "Install Xcode + CLT from App Store / xcode-select --install")

def unity_env():
    # עדיף Unity Hub CLI, אבל גם Unity Editor CLI תקף
    if shutil.which("unity") is None and shutil.which("Unity") is None and shutil.which("Unity.exe") is None:
        raise ResourceRequired("Unity CLI", "Install Unity/Hub and expose 'unity' CLI.")

def cuda_env():
    _require("nvidia-smi", "Install NVIDIA drivers/CUDA toolkit. For containers, use nvidia-container-runtime.")

def k8s_env():
    _require("kubectl", "Install kubectl; configure KUBECONFIG or in-cluster auth.")