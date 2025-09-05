# adapters/ios/sign.py
# -*- coding: utf-8 -*-
import os, subprocess, shlex
from ..contracts import ResourceRequired

def codesign_app(app_bundle_path: str, identity: str, entitlements_plist: str = None):
    """
    macOS codesign. אם חסר – ResourceRequired.
    """
    if not os.path.exists("/usr/bin/codesign"):
        raise ResourceRequired("Apple codesign", "Install Xcode Command Line Tools (xcode-select --install)")

    cmd = f"/usr/bin/codesign -s {shlex.quote(identity)} --force --timestamp"
    if entitlements_plist:
        cmd += f" --entitlements {shlex.quote(entitlements_plist)}"
    cmd += f" {shlex.quote(app_bundle_path)}"
    subprocess.run(cmd, shell=True, check=True)
    return {"ok": True, "bundle": app_bundle_path, "signed": True}