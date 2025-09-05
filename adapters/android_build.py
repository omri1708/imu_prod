# adapters/android_build.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from contracts.base import ensure_tool, run_ok, Artifact, ResourceRequired
from provenance.store import ProvenanceStore

ANDROID_SDK_HINT = (
    "Android SDK tools required. Install Android Studio or sdkmanager.\n"
    "- Linux/macOS: https://developer.android.com/studio\n"
    "- Ensure ANDROID_HOME or ANDROID_SDK_ROOT is set and platform-tools on PATH."
)

def _ensure_android_env():
    # java
    ensure_tool("java", "Install JDK (Temurin/Adoptium) and ensure 'java' on PATH.")
    # adb optional for signing/align; gradle will be used from wrapper if exists
    # sdk
    sdk = os.environ.get("ANDROID_SDK_ROOT") or os.environ.get("ANDROID_HOME")
    if not sdk or not Path(sdk).exists():
        raise ResourceRequired("Android SDK", ANDROID_SDK_HINT)

def build_gradle(project_dir: str, task: str = "assembleRelease", store: Optional[ProvenanceStore]=None) -> Artifact:
    """
    בונה אפליקציית אנדרואיד עם Gradle Wrapper אם קיים, אחרת עם gradle מהמחשב.
    """
    _ensure_android_env()
    p = Path(project_dir).resolve()
    if not (p / "app").exists():
        raise FileNotFoundError("expected Android project with app/ module")
    # gradle wrapper if present
    gw = "./gradlew" if (p / "gradlew").exists() else None
    if gw:
        run_ok([gw, task], cwd=str(p))
    else:
        ensure_tool("gradle", "Install Gradle or use Gradle Wrapper in the project.")
        run_ok(["gradle", task], cwd=str(p))

    # locate artifact (APK/AAB)
    apk = next((p / "app" / "build" / "outputs").rglob("*.apk"), None)
    aab = next((p / "app" / "build" / "outputs").rglob("*.aab"), None)
    if apk:
        art = Artifact(path=str(apk), kind="apk")
    elif aab:
        art = Artifact(path=str(aab), kind="aab")
    else:
        raise FileNotFoundError("no_apk_or_aab_found_in_outputs")

    if store:
        art = store.add(art, trust_level="built-local", evidence={"builder": "gradle"})
    return art