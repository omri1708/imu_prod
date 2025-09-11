# adapters/ios/build.py
# -*- coding: utf-8 -*-
import os
from typing import Dict, Any
import subprocess, shlex

from ..contracts import ensure_tool, run, record_provenance
from contracts.adapters import ios_env
from provenance.audit import AuditLog

def run_ios_build(cfg: Dict[str,Any], audit: AuditLog):
    ws = cfg["workspace"]
    scheme = cfg["scheme"]
    archive = cfg["archive_path"]
    export_path = cfg["export_path"]
    export_plist = cfg.get("export_options_plist")
    cmd1 = f'xcodebuild -workspace {shlex.quote(ws)} -scheme {shlex.quote(scheme)} -configuration Release -archivePath {shlex.quote(archive)} archive -allowProvisioningUpdates'
    cmd2 = f'xcodebuild -exportArchive -archivePath {shlex.quote(archive)} -exportPath {shlex.quote(export_path)}'
    if export_plist:
        cmd2 += f' -exportOptionsPlist {shlex.quote(export_plist)}'
    audit.append("adapter.ios","invoke",{"archive_cmd":cmd1,"export_cmd":cmd2})
    subprocess.check_call(cmd1, shell=True)
    subprocess.check_call(cmd2, shell=True)
    audit.append("adapter.ios","success",{"ipa_dir":export_path})
    return {"ok": True, "artifact_hint": export_path}


def build_xcarchive(project: str, scheme: str, out_dir: str) -> str:
    ios_env()
    os.makedirs(out_dir, exist_ok=True)
    archive = os.path.join(out_dir, f"{scheme}.xcarchive")
    cmd = f"xcodebuild -project {shlex.quote(project)} -scheme {shlex.quote(scheme)} -configuration Release -archivePath {shlex.quote(archive)} archive"
    subprocess.check_call(cmd, shell=True)
    return archive


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