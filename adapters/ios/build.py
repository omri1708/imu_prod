# adapters/ios/build.py
# -*- coding: utf-8 -*-
import os
from ..contracts import ensure_tool, run, record_provenance

def build_ios(project_dir: str, scheme: str, sdk: str = "iphonesimulator", configuration: str = "Debug") -> dict:
    """
    בונה .app או .ipa באמצעות xcodebuild (דורש macOS + Xcode מותקן).
    """
    ensure_tool("xcodebuild", "Install Xcode (App Store) and command line tools")
    derived = os.path.join(project_dir, "DerivedData")
    cmd = ["xcodebuild", "-scheme", scheme, "-sdk", sdk, "-configuration", configuration, "build", f"SYMROOT={derived}"]
    out = run(cmd, cwd=project_dir)
    # איתור תוצר
    app_path = None
    for root, _, files in os.walk(derived):
        for f in files:
            if f.endswith(".app"):
                app_path = os.path.join(root, f)
    if not app_path:
        raise RuntimeError("iOS app not found after build")
    prov = record_provenance("ios_build", {"project": project_dir, "scheme": scheme}, app_path)
    return {"app": app_path, "provenance": prov.__dict__, "log": out}