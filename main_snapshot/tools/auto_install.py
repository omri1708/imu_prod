# tools/auto_install.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, shutil, asyncio, json
from typing import List, Dict, Any, Tuple

def _which(x:str)->bool: return shutil.which(x) is not None
def _is_mac()->bool:     return sys.platform.startswith("darwin")
def _is_linux()->bool:   return sys.platform.startswith("linux")
def _is_win()->bool:     return sys.platform.startswith("win")

def detect_mgr()->Tuple[str,str]:
    """
    החזר (mgr, cmd) לפי זמין בסביבה:
      brew | apt-get | dnf | yum | winget | choco | pip
    """
    if _is_mac() and _which("brew"):     return ("brew", "brew")
    if _is_linux() and _which("apt-get"):return ("apt",  "sudo -n apt-get")
    if _is_linux() and _which("dnf"):    return ("dnf",  "sudo -n dnf")
    if _is_linux() and _which("yum"):    return ("yum",  "sudo -n yum")
    if _is_win() and _which("winget"):   return ("winget","winget")
    if _is_win() and _which("choco"):    return ("choco","choco")
    return ("none","")

def _pkg_for(check:str, mgr:str)->List[str]:
    """
    ממפה check -> פקודות התקנה לכל מנהל.
    שים לב: חלק מהדברים (Xcode/Unity) דורשים אינטראקציה—נחזיר “not_supported”.
    """
    m: Dict[str,Dict[str,List[str]]] = {
      "exe:node": {
        "brew":   ["brew install node"],
        "apt":    ["sudo -n apt-get update", "sudo -n apt-get install -y nodejs npm"],
        "dnf":    ["sudo -n dnf install -y nodejs npm"],
        "yum":    ["sudo -n yum install -y nodejs npm"],
        "winget": ["winget install --id OpenJS.NodeJS -e --source winget"],
        "choco":  ["choco install -y nodejs"],
      },
      "exe:npm": {  # בא עם node
        "brew":   ["brew install node"],
        "apt":    ["sudo -n apt-get install -y npm"],
        "dnf":    ["sudo -n dnf install -y npm"],
        "yum":    ["sudo -n yum install -y npm"],
        "winget": ["winget install --id OpenJS.NodeJS -e --source winget"],
        "choco":  ["choco install -y nodejs"],
      },
      "exe:gradle": {
        "brew":   ["brew install gradle"],
        "apt":    ["sudo -n apt-get install -y gradle"],
        "dnf":    ["sudo -n dnf install -y gradle"],
        "yum":    ["sudo -n yum install -y gradle"],
        "winget": ["winget install --id gradle.gradle -e"],
        "choco":  ["choco install -y gradle"],
      },
      "exe:sdkmanager": {  # Mac: Command Line Tools של אנדרואיד
        "brew":   ["brew install --cask android-commandlinetools"],
        "apt":    ["sudo -n apt-get install -y android-sdk || true"],
        "dnf":    ["sudo -n dnf install -y android-tools || true"],
        "yum":    ["sudo -n yum install -y android-tools || true"],
        "winget": ["winget install --id Google.AndroidSDK -e || true"],
        "choco":  ["choco install -y android-sdk || true"],
      },
      "exe:xcodebuild": {
        # דורש App Store/EULA – אי־אפשר סקריפטיבי
        "brew":   ["# not_supported: install Xcode from App Store"],
      },
      "exe:Unity": {
        "brew":   ["brew install --cask unity-hub || true"],  # רוב הסיכויים ידרוש GUI/EULA
        "winget": ["winget install UnityTechnologies.UnityHub || true"],
        "choco":  ["choco install -y unityhub || true"],
      },
      "exe:ffmpeg": {
        "brew":   ["brew install ffmpeg"],
        "apt":    ["sudo -n apt-get install -y ffmpeg"],
        "dnf":    ["sudo -n dnf install -y ffmpeg"],
        "yum":    ["sudo -n yum install -y ffmpeg"],
        "winget": ["winget install --id Gyan.FFmpeg -e"],
        "choco":  ["choco install -y ffmpeg"],
      },
      "exe:nvcc": {
        "brew":   ["# not_supported: CUDA Toolkit on macOS unsupported"],
        "apt":    ["# please install CUDA Toolkit from NVIDIA repo"],
        "dnf":    ["# please install CUDA Toolkit from NVIDIA repo"],
        "yum":    ["# please install CUDA Toolkit from NVIDIA repo"],
      },
      "exe:kubectl": {
        "brew":   ["brew install kubectl || brew install kubernetes-cli"],
        "apt":    ["sudo -n apt-get install -y kubectl || sudo -n snap install kubectl --classic || true"],
        "dnf":    ["sudo -n dnf install -y kubernetes-client || true"],
        "yum":    ["sudo -n yum install -y kubernetes-client || true"],
        "winget": ["winget install --id Kubernetes.kubectl -e"],
        "choco":  ["choco install -y kubernetes-cli"],
      },
      "exe:helm": {
        "brew":   ["brew install helm"],
        "apt":    ["sudo -n snap install helm --classic || true"],
        "dnf":    ["sudo -n dnf install -y helm || true"],
        "yum":    ["sudo -n yum install -y helm || true"],
        "winget": ["winget install --id Helm.Helm -e || true"],
        "choco":  ["choco install -y kubernetes-helm || true"],
      },
      "exe:sqlite3": {
        "brew":   ["brew install sqlite"],
        "apt":    ["sudo -n apt-get install -y sqlite3"],
        "dnf":    ["sudo -n dnf install -y sqlite || sudo -n dnf install -y sqlite3 || true"],
        "yum":    ["sudo -n yum install -y sqlite || true"],
        "winget": ["winget install --id SQLite.SQLite -e || true"],
        "choco":  ["choco install -y sqlite || true"],
      },
      "py:torch": {
        "pip":    ["python -m pip install --upgrade pip",
                   # ברירת מחדל CPU כדי לא להסתבך עם CUDA
                   "python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu"],
      },
      "py:pytest": {
        "pip":    ["python -m pip install --upgrade pip", "python -m pip install pytest"],
      },
    }
    mm = m.get(check, {})
    if mgr in mm:
        return mm[mgr]
    # pip- בלבד למודולים
    if check.startswith("py:"):
        return mm.get("pip", [])
    # אין מפה – נחזיר ריק
    return []

async def _run(cmd:str)->Tuple[int,str]:
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await proc.communicate()
    return proc.returncode, (out or b"").decode("utf-8","ignore")

async def auto_install_missing(missing: List[str]) -> Dict[str,Any]:
    """
    מתקין ברשיון “מה שאפשר עכשיו”, מחזיר:
      {"log":[{"check":..,"cmd":..,"rc":..,"out":..},...]}
    לא מפסיק על כשל; מנסה הכל, ואז ה־caller עושה rescan.
    """
    log: List[Dict[str,Any]] = []
    mgr, base = detect_mgr()
    for check in missing:
        cmds: List[str] = []
        # מודולי Python – תמיד נתקין ב-pip
        if check.startswith("py:"):
            cmds = _pkg_for(check, "pip")
        else:
            if mgr != "none":
                cmds = _pkg_for(check, mgr)
        if not cmds:
            log.append({"check":check,"cmd":"(no-automatic-install)","rc":127,"out":"no command for platform/manager"})
            continue
        for raw in cmds:
            if raw.strip().startswith("#"):
                log.append({"check":check,"cmd":raw,"rc":125,"out":"not_supported_or_manual"})
                continue
            rc, out = await _run(raw)
            log.append({"check":check,"cmd":raw,"rc":rc,"out":out})
    return {"log": log, "manager": mgr}