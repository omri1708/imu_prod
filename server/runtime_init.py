from __future__ import annotations
import os

def ensure_runtime_dirs():
    for d in ("var/llm_cache", "var/snapshots", "var/cas", "var/registry"):
        os.makedirs(d, exist_ok=True)
        keep = os.path.join(d, ".keep")
        try:
            open(keep, "a").close()
        except Exception:
            pass
    # probe write
    # per-run audit files are created under .imu/runs/<run_id>/audit by the run_context
    return True