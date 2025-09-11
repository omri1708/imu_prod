# server/planning.py
from __future__ import annotations
from typing import List, Dict, Any
try:
    from pydantic import BaseModel, Field  # v1
except Exception:
    from pydantic.v1 import BaseModel, Field

class BuildSpec(BaseModel):
    name: str = "app"
    goal: str
    stack: str = "fastapi+postgres+react"   # דוגמת ברירת מחדל
    features: List[str] = []
    non_goals: List[str] = []
    constraints: Dict[str, Any] = {}

def plan_from_text(llm, user_text: str, profile_hint: str="") -> BuildSpec:
    sys = ("את/ה מתכננ/ת תוכנה זהיר/ה. "
           "הפק פלט JSON תקין בלבד לשדות: name, goal, stack, features[], non_goals[], constraints{...}. "
           "אל תבטיח פעולה שלא ניתן לגבות.")
    content = f"פרופיל משתמש:\n{profile_hint}\n\nבקשת משתמש:\n{user_text}\n\nהחזר JSON תקין בלבד."
    draft = llm.chat([{"role":"system","content":sys},{"role":"user","content":content}],
                     temperature=0.2, max_tokens=700)
    # נפענח בזהירות; אם נכשל – נ fallback למינימום
    import json
    try:
        data = json.loads(draft)
        return BuildSpec(**data)
    except Exception:
        return BuildSpec(goal=user_text, features=[], non_goals=[], constraints={})
