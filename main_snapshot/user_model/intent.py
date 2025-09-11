# imu_repo/user_model/intent.py
from __future__ import annotations
from typing import Dict, Any, List

KEYS = {
    "realtime":  ["realtime", "real-time", "rt", "stream", "websocket", "webrtc", "low-latency"],
    "batch":     ["batch", "etl", "offline", "cron", "pipeline"],
    "mobile":    ["mobile", "android", "ios", "swiftui", "kotlin", "react-native"],
    "sensitive": ["pii", "secret", "secrets", "privacy", "gdpr", "hipaa", "sensitive"],
    "cost_saver":["cost", "optimize cost", "cheap", "low cost", "budget"],
    "gpu":       ["gpu", "cuda", "tensor", "ml", "inference"],
    "ui":        ["ui", "frontend", "react", "vue", "svelte", "unity"],
}

def infer_intent(spec: Dict[str,Any]) -> List[str]:
    text = (str(spec.get("name","")) + " " + str(spec.get("goal",""))).lower()
    tags: List[str] = []
    for tag, keys in KEYS.items():
        if any(k in text for k in keys):
            tags.append(tag)
    # ברירת מחדל—אם לא זוהה דבר: batch קל
    if not tags:
        tags.append("batch")
    return tags