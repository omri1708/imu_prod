# PATH: engine/diagnostics/traceback_analyzer.py
from __future__ import annotations
import re
from typing import Dict, Any

def parse_traceback(tb: str) -> Dict[str, Any]:
    """
    Very pragmatic traceback parser:
    - inspects the last exception type, message and most-recent frames
    - returns {category, message, path?, actions: [ {type, params} ], raw}
    """
    tb = tb or ""
    last = tb.strip().splitlines()[-1] if tb else ""
    cat, msg, path = "unknown", last, None
    actions = []

    # PermissionError: '/some/path'
    m = re.search(r"PermissionError.*?:.*?:\s*'([^']+)'", last)
    if m:
        cat = "permission"
        path = m.group(1)
        # הצעת תיקון: העבר נתיבי כתיבה ל-/tmp/imu_* (לוקאלי) או ל-/data/* (דוקר)
        actions.append({"type":"set_env", "key":"IMU_PROV_ROOT", "val":"/tmp/imu_provenance"})
        actions.append({"type":"set_env", "key":"IMU_STATE_DIR","val":"/tmp/imu_state"})
        actions.append({"type":"set_env", "key":"IMU_KEYS_PATH", "val":"/tmp/imu_keys"})
        return {"category":cat, "message":msg, "path":path, "actions":actions, "raw":tb}

    # ModuleNotFoundError: No module named 'bwrap.core'
    if "ModuleNotFoundError" in last and "bwrap.core" in last:
        cat = "sandbox_runner_bwrap"
        actions.append({"type":"set_env","key":"IMU_SANDBOX","val":"0"})
        actions.append({"type":"set_env","key":"IMU_BUILD_SANDBOX","val":"none"})
        actions.append({"type":"set_env","key":"IMU_BUILD_RUNNER","val":"direct"})
        actions.append({"type":"remove_bin","name":"bwrap"})
        return {"category":cat, "message":msg, "actions":actions, "raw":tb}

    # planner_empty_spec (ValidationFailed)
    if "planner_empty_spec" in tb:
        cat = "planner_empty_spec"
        actions.append({"type":"planner_fallback","mode":"intent_or_minimal"})
        return {"category":cat, "message":msg, "actions":actions, "raw":tb}

    # Generic ModuleNotFoundError: No module named 'X'
    m = re.search(r"ModuleNotFoundError: No module named '([^']+)'", last)
    if m:
        missing = m.group(1)
        cat="missing_module"
        # אם זה מודול פייתון בפרויקט שנוצר: הוסף ל-requirements.txt
        actions.append({"type":"ensure_requirement","package":missing})
        return {"category":cat, "message":msg, "module":missing, "actions":actions, "raw":tb}

    return {"category":cat, "message":msg, "actions":actions, "raw":tb}
