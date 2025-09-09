# engine/cleanup/gc.py
from __future__ import annotations
import os, time, glob

ROOTS = ["var/llm_cache", "var/audit", "var/snapshots"]

def gc_sweep(*, days: int = 7, max_file_mb: int = 200) -> None:
    cutoff = time.time() - days*86400
    for root in ROOTS:
        if not os.path.exists(root): continue
        for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            try:
                if os.path.isdir(p):
                    continue
                if os.path.getmtime(p) < cutoff:
                    os.remove(p); continue
                if os.path.getsize(p) > max_file_mb*1024*1024:
                    # רוטציה פשוטה
                    os.rename(p, p+".old")
            except Exception:
                pass