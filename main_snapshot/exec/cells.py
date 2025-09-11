# imu_repo/exec/cells.py
from __future__ import annotations
import os, time
from typing import Dict, Any
from exec.errors import ResourceRequired, ExecError
from exec.languages import python_runner, node_runner, go_runner, java_runner, csharp_runner, cpp_runner, rust_runner

RUNNERS = {
    "python": python_runner.run,
    "node": node_runner.run,
    "go": go_runner.run,
    "java": java_runner.run,
    "csharp": csharp_runner.run,
    "cpp": cpp_runner.run,
    "rust": rust_runner.run,
}

def run_code(lang: str, code: str, user_id: str = "anon", cell_name: str = "cell") -> Dict[str,Any]:
    lang = lang.lower()
    if lang not in RUNNERS: raise ExecError(f"unsupported_lang:{lang}")
    root = os.path.join(".imu_state","cells", user_id, lang, f"{int(time.time()*1000)}_{cell_name}")
    res = RUNNERS[lang](code, root)
    # מטא בסיסי
    res["workdir"] = root
    return res