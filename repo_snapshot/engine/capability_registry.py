# engine/capability_registry.py
REGISTRY = {
    "android": {"desc": "Build Android apps via Gradle"},
    "ios": {"desc": "Build iOS apps via xcodebuild"},
    "unity": {"desc": "Unity CLI batchmode"},
    "cuda": {"desc": "CUDA job runner (requires NVIDIA runtime)"},
    "k8s": {"desc": "Kubernetes job/plugin executor"},
}