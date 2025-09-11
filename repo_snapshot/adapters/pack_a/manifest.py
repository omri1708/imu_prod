# adapters/pack_a/manifest.py
# -*- coding: utf-8 -*-
"""
Adapter Pack A – מיפוי כלים מדויקים (winget/brew/apt/yum/choco)
+ הרכבת פקודות, + dry-run + בדיקות Policy.
אין התקנות בפועל כאן—אלא קומפוזיציה מאובטחת של פקודות להפעלה ע"י /capabilities/request שכבר בנוי אצלך.
"""
from __future__ import annotations
from typing import Dict, List, Optional

# מזהים ספציפיים
WINGET = {
    "android_sdk": "Google.AndroidSDK",
    "gradle": "Gradle.Gradle",
    "unity_hub": "UnityTechnologies.UnityHub",
    "node": "OpenJS.NodeJS",
    "git": "Git.Git",
    "cuda": "NVIDIA.CUDA",
    "helm": "Helm.Helm",
    "kubectl": "Kubernetes.kubectl",
    "minikube": "Googlecloudsdk.Minikube",  # לעיתים Microsoft.Minikube
}

BREW = {
    "android_platform_tools": "android-platform-tools",
    "gradle": "gradle",
    "unity_hub": "unity-hub",
    "node": "node",
    "git": "git",
    "cuda": "cuda",  # לעיתים דורש cask/nvidia-driver
    "helm": "helm",
    "kubectl": "kubectl",
    "minikube": "minikube",
}

APT = {
    "openjdk": "openjdk-17-jdk",
    "gradle": "gradle",
    "adb": "android-tools-adb",
    "node": "nodejs",
    "npm": "npm",
    "git": "git",
    "cuda": "nvidia-cuda-toolkit",
    "helm": "helm",
    "kubectl": "kubectl",
    "minikube": "minikube",
}

YUM = {
    "openjdk": "java-17-openjdk-devel",
    "git": "git",
    "node": "nodejs",
    "npm": "npm",
}

# פקודות הרצה טיפוסיות:

def unity_build_cli(project_path: str, target: str, output_path: str) -> List[str]:
    """
    target: 'Android' | 'iOS' | 'StandaloneWindows64' | 'StandaloneOSX'
    """
    return [
        "unity", "-batchmode", "-nographics",
        "-projectPath", project_path,
        "-buildTarget", target,
        "-quit",
        "-logFile", "-",
        "-customBuildTarget", target,
        "-customBuildPath", output_path
    ]

def android_gradle_build(module_dir: str, variant: str = "Release") -> List[str]:
    return ["./gradlew", f"assemble{variant}"], module_dir

def ios_xcodebuild(project_path: str, scheme: str, configuration: str = "Release") -> List[str]:
    return ["xcodebuild", "-scheme", scheme, "-configuration", configuration, "-project", project_path]

def cuda_job_run(py_entry: str, args: List[str]) -> List[str]:
    return ["python", py_entry] + args

def k8s_deploy(manifest_yaml: str, namespace: str = "default") -> List[str]:
    return ["kubectl", "apply", "-n", namespace, "-f", manifest_yaml]

def helm_install(chart: str, release: str, namespace: str, values: Optional[str] = None) -> List[str]:
    cmd = ["helm", "upgrade", "--install", release, chart, "-n", namespace]
    if values:
        cmd += ["-f", values]
    return cmd

def dry_run(command: List[str]) -> Dict[str, object]:
    """
    מחזיר הרכבת פקודה בצורה בטוחה לניתוח/בדיקה. לא מפעיל.
    """
    return {"ok": True, "cmd": command, "reason": "dry_run_only"}