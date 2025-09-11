# demos/android_ios_build.py
from __future__ import annotations
import os, subprocess, shutil, tempfile
from typing import Optional
from adapters.installer import ensure
from common.provenance import CAS
from security.policy import UserPolicy, check_fs, record_audit

def run(cmd, cwd=None):
    print("+"," ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

def ensure_android_env():
    # נדרש: java, gradle, sdkmanager
    ensure("javac","jdk")
    ensure("gradle","gradle")
    # sdkmanager מגיע עם cmdline-tools; נבדוק:
    if not shutil.which("sdkmanager"):
        raise RuntimeError("Android cmdline-tools (sdkmanager) missing")

def gradle_build(app_dir:str, flavor:Optional[str]=None, task="assembleRelease")->str:
    args=["gradle", task]
    if flavor: args=["gradle", f"assemble{flavor.capitalize()}Release"]
    run(args, cwd=app_dir)
    # מצא APK
    apk=None
    for root,_,files in os.walk(os.path.join(app_dir,"app","build","outputs")):
        for f in files:
            if f.endswith(".apk"):
                apk=os.path.join(root,f)
    if not apk: raise RuntimeError("APK not found after build")
    return apk

def ensure_ios_env():
    # נדרש Xcode CLI tools
    if not shutil.which("xcodebuild"):
        raise RuntimeError("xcodebuild missing (install Xcode Command Line Tools)")

def xcode_build(project_path:str, scheme:str, configuration="Release")->str:
    run(["xcodebuild","-scheme",scheme,"-configuration",configuration,"build","CODE_SIGNING_ALLOWED=NO"], cwd=project_path)
    # מצא .app או .ipa (בפועל נדרשת חתימה; כאן איבוד חתימה לצורך הדגמה, CAS ישמור)
    out=None
    for root,_,files in os.walk(project_path):
        for f in files:
            if f.endswith(".app") or f.endswith(".ipa"):
                out=os.path.join(root,f)
    if not out: raise RuntimeError("iOS artifact not found")
    return out

def publish_cas(path:str, trust="medium"):
    cas=CAS("cas")
    ev=cas.put(path)
    # חתימה דמה (ניתן להרחיב ל־codesign/gpg/sigstore)
    ev.trust = trust
    cas._write_meta(ev)
    return ev

def main_android(app_dir:str, user="default"):
    pol=UserPolicy(user_id=user)
    check_fs(pol,"read", app_dir)
    ensure_android_env()
    apk=gradle_build(app_dir)
    ev=publish_cas(apk, trust="high")
    record_audit("android_build", user, {"apk": ev.sha256, "path": ev.path})
    print("APK published to CAS:", ev.sha256)

def main_ios(project_dir:str, scheme:str, user="default"):
    pol=UserPolicy(user_id=user)
    check_fs(pol,"read", project_dir)
    ensure_ios_env()
    art=xcode_build(project_dir, scheme)
    ev=publish_cas(art, trust="high")
    record_audit("ios_build", user, {"artifact": ev.sha256, "path": ev.path})
    print("iOS artifact published to CAS:", ev.sha256)

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("usage: android_ios_build.py android <app_dir> | ios <xcodeproj_dir> <scheme>")
        raise SystemExit(2)
    if sys.argv[1]=="android":
        main_android(sys.argv[2])
    elif sys.argv[1]=="ios":
        main_ios(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(2)