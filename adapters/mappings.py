# adapters/mappings.py
# מיפויי חבילות מדויקים + תבניות CLI דטרמיניסטיות לכל Adapter

from __future__ import annotations

WINGET = {
    "unity.hub": "UnityTechnologies.UnityHub",
    "android.sdk": "Google.AndroidSDK",
    "jdk": "EclipseAdoptium.Temurin.17.JDK",
    "gradle": "Gradle.Gradle",
    "nodejs": "OpenJS.NodeJS.LTS",
    "kubectl": "Kubernetes.kubectl",
    "docker": "Docker.DockerDesktop",
    "cuda": "Nvidia.CUDA",
}

BREW = {
    "unity.hub": "unity-hub",
    "android.sdk": "android-sdk",   # cask בסביבות מסוימות
    "jdk": "temurin@17",
    "gradle": "gradle",
    "nodejs": "node",               # LTS בפועל
    "kubectl": "kubectl",
    "docker": "colima",             # או Docker Desktop לפי העדפה
    "cuda": "cuda",
}

APT = {
    "jdk": "openjdk-17-jdk",
    "gradle": "gradle",
    "nodejs": "nodejs",
    "npm": "npm",
    "kubectl": "kubectl",
    "docker": "docker.io",
    "cuda": "nvidia-cuda-toolkit",
}

# תבניות CLI (מפתחות params חייבים להתקבל ב-/adapters/dry_run)
CLI_TEMPLATES = {
    "unity.build": {
        "linux": "/opt/unity/Editor/Unity -batchmode -nographics -quit -projectPath {project} -buildTarget {target} -executeMethod {method} -logFile {log}",
        "mac": "/Applications/Unity/Hub/Editor/{version}/Unity.app/Contents/MacOS/Unity -batchmode -projectPath {project} -buildTarget {target} -executeMethod {method} -logFile {log} -quit",
        "win": "C:\\Program Files\\Unity\\Hub\\Editor\\{version}\\Editor\\Unity.exe -batchmode -projectPath {project} -buildTarget {target} -executeMethod {method} -logFile {log} -quit",
    },
    "android.gradle": {
        "any": "./gradlew assemble{flavor}{buildType} -Pandroid.injected.signing.store.file={keystore}"
    },
    "ios.xcode": {
        "mac": "xcodebuild -workspace {workspace} -scheme {scheme} -configuration {config} -destination 'generic/platform=iOS' build"
    },
    "k8s.kubectl.apply": {
        "any": "kubectl apply -f {manifest} --namespace {namespace}"
    },
    "cuda.nvcc": {
        "any": "nvcc {src} -o {out}"
    },
}