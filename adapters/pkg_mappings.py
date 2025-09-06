# adapters/pkg_mappings.py
from __future__ import annotations

# מיפויים מדויקים/ריאליים ככל האפשר לשמות חבילות בכל מנהל־חבילות
WINGET = {
    "git":          "Git.Git",
    "python":       "Python.Python.3.12",
    "node":         "OpenJS.NodeJS",
    "ffmpeg":       "Gyan.FFmpeg",
    "kubectl":      "Kubernetes.kubectl",
    "helm":         "Helm.Helm",
    "unity.hub":    "UnityTechnologies.UnityHub",
    # Unity editor versions עדיף דרך hub cli בפרויקט עצמו
    "android.sdk":  "Google.AndroidSDK",
    "gradle":       "Gradle.Gradle",
    "jdk":          "EclipseAdoptium.Temurin.17.JDK",
    "cuda":         "Nvidia.CUDA",  # יתכן דרוש reboot/driver
}

BREW = {
    "git": "git",
    "python": "python@3.12",
    "node": "node",
    "ffmpeg": "ffmpeg",
    "kubectl": "kubectl",
    "helm": "helm",
    "gradle":"gradle",
    "android.sdk":"android-sdk",      # ב־homebrew-cask
    "unity.hub":"unity-hub",          # cask
    "jdk":"temurin@17",               # או openjdk@17
    "cuda":"cuda",                    # דורש חומרה/דרייבר
}

APT = {
    "git": "git",
    "python": "python3 python3-venv python3-pip",
    "node": "nodejs npm",
    "ffmpeg": "ffmpeg",
    "kubectl": "kubectl",  # לרוב דרך repo הרשמי של k8s
    "helm": "helm",
    "gradle":"gradle",
    "jdk":"openjdk-17-jdk",
    "android.sdk":"cmdline-tools",   # לרוב דרך sdkmanager אחרי בסיסי
    "cuda":"nvidia-cuda-toolkit",
}