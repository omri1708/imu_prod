# adapters/ios.py
# -*- coding: utf-8 -*-
import os, subprocess, shutil
from typing import Tuple, List
from policy.policy_engine import PolicyViolation

def dry_run(xcodeproj_path:str)->Tuple[bool,List[str]]:
    missing=[]
    if not shutil.which("xcodebuild"):
        missing.append("Xcode not found (requires macOS)")
    if not os.path.exists(xcodeproj_path):
        missing.append("xcodeproj not found")
    return (len(missing)==0, missing)

def build_ipa(xcodeproj_path:str, scheme:str, out_dir:str)->str:
    ok, why = dry_run(xcodeproj_path)
    if not ok: raise PolicyViolation("ios dry-run failed: " + "; ".join(why))
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["xcodebuild","-project",xcodeproj_path,"-scheme",scheme,"-configuration","Release","archive",
           "-archivePath","build/app.xcarchive","CODE_SIGN_STYLE=Automatic"]
    subprocess.run(cmd, check=True)
    cmd = ["xcodebuild","-exportArchive","-archivePath","build/app.xcarchive",
           "-exportOptionsPlist","ExportOptions.plist","-exportPath",out_dir]
    subprocess.run(cmd, check=True)
    # locate IPA
    for f in os.listdir(out_dir):
        if f.endswith(".ipa"):
            return os.path.join(out_dir,f)
    raise RuntimeError("ipa not found")