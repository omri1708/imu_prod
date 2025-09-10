# PATH: engine/self_heal/auto_fix.py
from __future__ import annotations
import os, io
from typing import Dict, Any

def apply_actions(actions: list[dict], files: Dict[str, bytes|str] | None = None) -> Dict[str, Any]:
    """Apply simple actions to current process env and generated files."""
    result = {"applied": [], "errors": []}
    files = files or {}

    for act in actions:
        t = act.get("type")
        try:
            if t == "set_env":
                os.environ[str(act["key"])] = str(act["val"])
                result["applied"].append(act)

            elif t == "remove_bin":
                # best-effort: אם יש סקריפט bwrap ב-venv – ננטרל
                name = str(act.get("name",""))
                for p in (".venv/bin", "venv/bin", "bin"):
                    fp = f"{p}/{name}"
                    if os.path.exists(fp):
                        try:
                            os.remove(fp)
                            result["applied"].append({"type":"remove","path":fp})
                        except Exception as e:
                            result["errors"].append({"action":act,"error":str(e)})

            elif t == "ensure_requirement":
                pkg = str(act.get("package"))
                # אם קיימת מפת קבצים גנרית – נעדכן את requirements בהם
                for req_path in ("services/api/requirements.txt","requirements.txt"):
                    if req_path in files:
                        text = files[req_path].decode() if isinstance(files[req_path], (bytes,bytearray)) else str(files[req_path])
                        if pkg not in text:
                            text += f"\n{pkg}\n"
                            files[req_path] = text.encode("utf-8")
                            result["applied"].append({"type":"patched_req","file":req_path,"pkg":pkg})

            elif t == "planner_fallback":
                # no-op here; הדבר מטופל בצד ה-router (נפילה ל-IntentToSpec/Minimal)
                result["applied"].append(act)

        except Exception as e:
            result["errors"].append({"action":act,"error":str(e)})

    return result
