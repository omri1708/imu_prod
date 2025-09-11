# knowledge/tools_store.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, shutil, importlib
from typing import Dict, Any, List

STATE = ".imu_state"
TOOLS_JSON = os.path.join(STATE, "tools.json")
DEFAULT_TTL = 24*3600  # יום

# בדיקות זמינות בסיסיות
CHECKS = {
    "exe:node":        lambda: shutil.which("node")      is not None,
    "exe:npm":         lambda: shutil.which("npm")       is not None,
    "exe:gradle":      lambda: shutil.which("gradle")    is not None,
    "exe:sdkmanager":  lambda: shutil.which("sdkmanager")is not None,
    "exe:xcodebuild":  lambda: shutil.which("xcodebuild")is not None,
    "exe:Unity":       lambda: shutil.which("Unity")     is not None or shutil.which("unity-editor") is not None,
    "exe:kubectl":     lambda: shutil.which("kubectl")   is not None,
    "exe:helm":        lambda: shutil.which("helm")      is not None,
    "exe:ffmpeg":      lambda: shutil.which("ffmpeg")    is not None,
    "exe:nvcc":        lambda: shutil.which("nvcc")      is not None,
    "exe:sqlite3":     lambda: shutil.which("sqlite3")   is not None,
    "py:insightface":  lambda: _py("insightface"),
    "py:torch":        lambda: _py("torch"),
}
def _py(mod:str)->bool:
    try: importlib.import_module(mod); return True
    except Exception: return False

def _load() -> Dict[str,Any]:
    if not os.path.exists(STATE): os.makedirs(STATE, exist_ok=True)
    if os.path.exists(TOOLS_JSON):
        try: return json.loads(open(TOOLS_JSON,"r",encoding="utf-8").read())
        except Exception: pass
    return {"items":{}, "ts": time.time(), "ttl": DEFAULT_TTL}

def _save(st:Dict[str,Any]):
    if not os.path.exists(STATE): os.makedirs(STATE, exist_ok=True)
    open(TOOLS_JSON,"w",encoding="utf-8").write(json.dumps(st, ensure_ascii=False, indent=2))

def snapshot(keys: List[str], *, force: bool=False, ttl:int=DEFAULT_TTL) -> Dict[str,bool]:
    st = _load()
    if force or (time.time() - st.get("ts",0) > st.get("ttl",ttl)):
        st["items"] = {}
    for k in keys:
        if k not in st["items"]:
            st["items"][k] = bool(CHECKS.get(k, lambda: False)())
    st["ts"] = time.time()
    st["ttl"] = ttl
    _save(st)
    return st["items"]

def get_all() -> Dict[str,bool]:
    return _load().get("items",{})

def remember(keys: List[str], values: Dict[str,bool]):
    st = _load()
    for k,v in values.items():
        st["items"][k] = bool(v)
    st["ts"] = time.time()
    _save(st)