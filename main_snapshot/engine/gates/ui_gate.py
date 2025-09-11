# imu_repo/engine/gates/ui_gate.py
from __future__ import annotations
from typing import Dict, Any
from ui.accessibility_gate import check_directory

class UIGate:
    """
    מפעיל בדיקות נגישות על תיקיית אתר שנוצרה ע"י ui/gen_frontend.py.
      cfg = {"path": "/mnt/data/imu_repo/site", "min_contrast": 4.5}
    """
    def __init__(self, path: str, *, min_contrast: float=4.5):
        self.path = path
        self.min_contrast = float(min_contrast)

    def check(self) -> Dict[str,Any]:
        res = check_directory(self.path, min_contrast=self.min_contrast)
        return res