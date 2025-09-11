# imu_repo/engine/metrics/jsonl.py
from pathlib import Path
import json
def append_jsonl(path: str, obj: dict) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
