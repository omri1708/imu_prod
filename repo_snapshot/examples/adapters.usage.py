from adapters.android.build import build_apk
from adapters.ios.build import build_xcarchive
from adapters.unity.cli import unity_batch
from adapters.cuda.runner import run_cuda_job
from adapters.k8s.deploy import canary_and_rollout
from engine.errors import ResourceRequired

def main():
    try:
        apk = build_apk("mobile/MyAndroidApp", "Release")
        print("Built APK:", apk)
    except ResourceRequired as e:
        print("Android build needs:", e)

    try:
        xc = build_xcarchive("ios/MyApp.xcodeproj", "MyApp", "dist")
        print("Built xcarchive:", xc)
    except ResourceRequired as e:
        print("iOS build needs:", e)

    try:
        unity_batch("unity/MyGame", "Builder.Perform")
        print("Unity batch ok")
    except ResourceRequired as e:
        print("Unity build needs:", e)

    try:
        run_cuda_job("jobs/cuda_task.py", ["--epochs","1"])
        print("CUDA job ok")
    except ResourceRequired as e:
        print("CUDA job needs:", e)

    try:
        canary_and_rollout("k8s/deploy.yaml", "k8s/canary.yaml", name="web", ns="prod")
        print("K8s deploy ok")
    except ResourceRequired as e:
        print("K8s needs:", e)

if __name__ == "__main__":
    main()