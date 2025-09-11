# imu_repo/capabilities/fs_read.py
from __future__ import annotations
import os
from typing import Dict, Any
from grounded.claims import current

async def read_text(spec: Dict[str,Any]) -> str:
    """
    spec = { "path": "/tmp/file.txt" }
    """
    path = str(spec["path"])
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    current().add_evidence("fs_read", {
        "source_url": f"file://{path}",
        "trust": 0.8,    # אמון סביר לקבצים מקומיים בטסט
        "ttl_s": 3600,
        "payload": {"bytes": len(data.encode('utf-8'))}
    })
    return data
