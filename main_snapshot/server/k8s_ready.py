# server/k8s_ready.py
# בדיקות readiness אמיתיות ב־Cluster בלי ספריות חיצוניות: kubectl get pods -o json
from __future__ import annotations
import subprocess, json, shutil
from typing import Dict, Any

def have_kubectl() -> bool:
    return shutil.which("kubectl") is not None

def _kubectl(args: list[str]) -> str:
    p = subprocess.run(["kubectl"]+args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout)
    return p.stdout

def readiness_ratio(namespace: str, app: str) -> Dict[str, Any]:
    """
    מחזיר {"ok": bool, "ratio": float, "pods": N, "ready": R} לפי מצב pods:
      - מסתכל על status.containerStatuses[].ready לכל קונטיינר
      - פוד נחשב ready רק אם כל הקונטיינרים ready ו־conditions Ready==True
    """
    out = _kubectl(["get","pods","-n",namespace,"-l",f"app={app}","-o","json"])
    j = json.loads(out)
    total = 0
    ready = 0
    details = []
    for pod in j.get("items", []):
        total += 1
        pod_name = pod["metadata"]["name"]
        # conditions
        cond_ok = any((c.get("type")=="Ready" and c.get("status")=="True") for c in pod.get("status",{}).get("conditions",[]))
        cs = pod.get("status",{}).get("containerStatuses",[])
        cs_ok = bool(cs) and all(c.get("ready") for c in cs)
        is_ready = cond_ok and cs_ok
        if is_ready: ready += 1
        details.append({"pod": pod_name, "ready": is_ready, "cond": cond_ok, "containers_ready": cs_ok})
    ratio = (ready / total) if total else 0.0
    return {"ok": True, "ratio": ratio, "pods": total, "ready": ready, "details": details}