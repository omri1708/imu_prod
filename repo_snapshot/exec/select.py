# imu_repo/exec/select.py
from __future__ import annotations
from typing import Dict, Any, List
from exec.detect import detect

# מפה פשוטה: tag -> עדיפות שפות
PREF = {
    "web":        ["node","go","python"],
    "numerics":   ["cpp","rust","python"],
    "system":     ["rust","cpp","go"],
    "scripting":  ["python","node"],
    "concurrency":["go","rust","cpp"],
    "ml":         ["python","cpp"],
    "enterprise": ["java","csharp","go"],
}

def choose(task_tags: List[str]) -> List[str]:
    tools = detect()
    scored = {}
    for tag in (task_tags or ["scripting"]):
        for i, lang in enumerate(PREF.get(tag, [])):
            if tools.get(_map_tool(lang)):  # זמין
                scored[lang] = min(scored.get(lang, 99), i)
    # ברירת מחדל: python אם קיים
    if not scored and tools.get(_map_tool("python")):
        return ["python"]
    # החזר לפי ציון
    return sorted(scored, key=lambda k: scored[k])

def _map_tool(lang: str) -> str:
    return {
        "python":"python",
        "node":"node",
        "go":"go",
        "java":"javac",
        "csharp":"dotnet",
        "cpp":"g++",
        "rust":"rustc",
    }[lang]