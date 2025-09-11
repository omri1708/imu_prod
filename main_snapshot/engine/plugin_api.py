# imu_repo/engine/plugin_api.py
from __future__ import annotations
from typing import Dict, Any, Protocol

class Plugin(Protocol):
    """
    כל תוסף מקבל:
      - spec: ה-BuildSpec (או dict עם extras)
      - build_dir: תיקיית ה-build לראיות/ארטיפקטים
      - user_id: מזהה משתמש
    ומחזיר dict evidence + KPI חלקי (אם רלוונטי).
    """
    def run(self, spec: Any, build_dir: str, user_id: str) -> Dict[str,Any]:
        ...