# adapters/installer.py
from __future__ import annotations
import os, subprocess, sys, shutil, platform
from typing import Optional, List
from adapters.pkg_mappings import WINGET, BREW, APT

class InstallError(RuntimeError): ...

def have(cmd:str)->bool:
    return shutil.which(cmd) is not None

def run(cmd:List[str], check=True)->int:
    print("+"," ".join(cmd))
    return subprocess.run(cmd, check=check).returncode

def request_capability(pkg_key:str)->bool:
    """מבקש להתקין חבילה לפי מערכת ההפעלה. מחזיר True אם הצליח/כבר מותקן."""
    system = platform.system().lower()
    if pkg_key=="unity.cli":
        # מניחים hub קיים; CLI של יוניטי מנוהל דרך Hub (unity-editor --headless וכו')
        return have("unity") or have("Unity") or have("unity-editor") or have("Unity Hub")
    if system=="windows":
        if not have("winget"): raise InstallError("winget not available on Windows")
        pkg = WINGET.get(pkg_key)
        if not pkg: raise InstallError(f"no winget mapping for {pkg_key}")
        try:
            run(["winget","install","-e","--id",pkg,"-h"])
            return True
        except Exception as e:
            print("winget install failed:", e); return False
    elif system=="darwin":
        # brew + --cask כשצריך
        if not have("brew"): raise InstallError("homebrew not installed")
        name = BREW.get(pkg_key)
        if not name: raise InstallError(f"no brew mapping for {pkg_key}")
        args=["brew","install"]
        if pkg_key in ("unity.hub","android.sdk"): args=["brew","install","--cask"]
        try:
            run(args+[name]); return True
        except Exception as e:
            print("brew install failed:", e); return False
    else:
        # linux
        if have("apt"):
            pkgs = APT.get(pkg_key)
            if not pkgs: raise InstallError(f"no apt mapping for {pkg_key}")
            try:
                run(["sudo","apt","update"])
                run(["sudo","apt","install","-y"]+pkgs.split())
                return True
            except Exception as e:
                print("apt install failed:", e); return False
        raise InstallError("unsupported linux package manager")

def ensure(cmd:str, pkg_hint:str)->None:
    if have(cmd): return
    ok = request_capability(pkg_hint)
    if not ok or not have(cmd):
        raise InstallError(f"required tool '{cmd}' unavailable; attempted install as '{pkg_hint}' and still missing")