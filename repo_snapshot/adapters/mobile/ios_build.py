#  adapters/mobile/ios_build.py
# -*- coding: utf-8 -*-
import os, shutil, subprocess
from adapters.contracts import ResourceRequired

def build_ios(project: str, scheme: str, sdk: str = "iphoneos", configuration: str = "Release", out_dir: str = "build_ios"):
    if not shutil.which("xcodebuild"):
        raise ResourceRequired("Xcode/xcodebuild", "Install Xcode command-line tools")
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["xcodebuild", "-project", project, "-scheme", scheme, "-sdk", sdk,
           "-configuration", configuration, "BUILD_DIR="+os.path.abspath(out_dir)]
    subprocess.run(cmd, check=True)
    return {"ok": True, "build_dir": out_dir}