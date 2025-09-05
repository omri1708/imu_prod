# examples/run_adapters.py
# -*- coding: utf-8 -*-
import os, json, sys
from adapters.contracts import ResourceRequired
from adapters.android.build import build_android
from adapters.ios.build import build_ios
from adapters.unity.cli_build import build_unity
from adapters.gpu.cuda_runner import compile_and_run_cuda
from adapters.k8s.deploy import deploy

def main():
    results = {}

    # ANDROID
    if len(sys.argv) > 1 and sys.argv[1] == "android":
        project = os.environ.get("ANDROID_PROJECT", "/path/to/android/project")
        try:
            r = build_android(project, variant="Debug")
            results["android"] = r
        except ResourceRequired as rr:
            print(f"[ANDROID] Missing: {rr.what}\nInstall hint: {rr.how_to_install}")
        except Exception as e:
            print(f"[ANDROID] Error: {e}")

    # iOS
    if len(sys.argv) > 1 and sys.argv[1] == "ios":
        project = os.environ.get("IOS_PROJECT", "/path/to/ios/project")
        scheme = os.environ.get("IOS_SCHEME", "App")
        try:
            r = build_ios(project, scheme=scheme, sdk="iphonesimulator", configuration="Debug")
            results["ios"] = r
        except ResourceRequired as rr:
            print(f"[iOS] Missing: {rr.what}\nInstall hint: {rr.how_to_install}")
        except Exception as e:
            print(f"[iOS] Error: {e}")

    # UNITY
    if len(sys.argv) > 1 and sys.argv[1] == "unity":
        unity_proj = os.environ.get("UNITY_PROJECT", "/path/to/unity/project")
        try:
            r = build_unity(unity_proj, build_target="StandaloneWindows64", output_path="Build/Game.exe")
            results["unity"] = r
        except ResourceRequired as rr:
            print(f"[UNITY] Missing: {rr.what}\nInstall hint: {rr.how_to_install}")
        except Exception as e:
            print(f"[UNITY] Error: {e}")

    # CUDA
    if len(sys.argv) > 1 and sys.argv[1] == "cuda":
        try:
            r = compile_and_run_cuda()
            results["cuda"] = r
        except ResourceRequired as rr:
            print(f"[CUDA] Missing: {rr.what}\nInstall hint: {rr.how_to_install}")
        except Exception as e:
            print(f"[CUDA] Error: {e}")

    # K8s
    if len(sys.argv) > 1 and sys.argv[1] == "k8s":
        try:
            r = deploy(name="imu-demo", image="nginx:alpine", port=8080, replicas=1)
            results["k8s"] = r
        except ResourceRequired as rr:
            print(f"[K8s] Missing: {rr.what}\nInstall hint: {rr.how_to_install}")
        except Exception as e:
            print(f"[K8s] Error: {e}")

    if results:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("Choose one: android | ios | unity | cuda | k8s")

if __name__ == "__main__":
    main()