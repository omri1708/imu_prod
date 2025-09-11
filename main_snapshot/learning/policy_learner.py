# -*- coding: utf-8 -*-
from __future__ import annotations
import os, yaml, time, json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from executor.policy import Policy
from assurance.errors import ResourceRequired
from capabilities.synth import CapabilitySynthesizer, CapabilitySpec

@dataclass
class MetricsSnapshot:
    total_commits: int = 0
    resource_required_counts: Dict[str,int] = None

class PolicyTuner:
    """
    לומד ומעדכן את executor/policy.yaml + יוזם יצירת adapters כש-Z מתבקש שוב ושוב.
    """
    def __init__(self, policy_path: str = "./executor/policy.yaml",
                 adapters_root: str = "./adapters/generated"):
        self.policy_path = policy_path
        self.adapters_root = adapters_root
        self._policy_cache: Optional[Policy] = None
        self._resource_counts: Dict[str,int] = {}
        self._last_saved = 0

    def load(self) -> Policy:
        self._policy_cache = Policy.load(self.policy_path)
        return self._policy_cache

    def _save_yaml(self, y: Dict[str,Any]):
        with open(self.policy_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(y, f, sort_keys=False, allow_unicode=True)

    def as_yaml(self) -> Dict[str,Any]:
        # קריאה של קובץ YAML גולמי למילון; לא נעזר במחלקת Policy לצורך שמירה
        return yaml.safe_load(open(self.policy_path,"r",encoding="utf-8"))

    def observe_resource_required(self, what: str):
        self._resource_counts[what] = 1 + self._resource_counts.get(what,0)

    def tune_from_metrics(self, m: MetricsSnapshot) -> Dict[str,Any]:
        """
        כללי למידה פשוטים (KISS):
        - אם יש הרבה timeouts/זיכרון? (נדרוש בעתיד ספירת שגיאות)—נעלה מעט wall/cpu/mem (עד גבול).
        - אם כלי חסר חוזר שוב ושוב: נוסיף אותו ל-allowed_tools עם args_regex=".*" ונצית auto-synth adapter.
        """
        y = self.as_yaml()
        changed = False

        # 1) כלי חסר שחוזר שוב ושוב -> הוספה ל-allowed_tools + synth adapter
        for k,c in list(self._resource_counts.items()):
            # תצורה: what="tool:xxx" או "cmd_not_found:xxx"
            name = None
            if k.startswith("tool:"): name = k.split(":",1)[1]
            elif k.startswith("cmd_not_found:"): name = k.split(":",1)[1]
            if not name: continue
            tools = y.get("allowed_tools") or []
            if all(t.get("name")!=name for t in tools) and c >= 3:
                tools.append({"name": name, "args_regex": ".*", "allow_net": False})
                y["allowed_tools"] = tools
                changed = True
                # צור adapter בסיסי (On-Demand)
                try:
                    syn = CapabilitySynthesizer(self.adapters_root)
                    syn.generate(CapabilitySpec(kind=f"tool.{name}", cmd=name, args_template="",
                                                schema={"type":"object","properties":{},"additionalProperties":True},
                                                install_hint=f"install '{name}' via system package manager"))
                except Exception:
                    pass

        # 2) העלאת תקציבים זהירה — (future work: כשנקלוט counters לריצות שכשלו בזמן/זיכרון)
        # כאן נשאיר KISS: לא נשנה בלי שיש לנו ספירות ברורות (נוסיף בהמשך כשנאסוף נתונים).

        if changed:
            self._save_yaml(y)
            self._last_saved = time.time()
        return {"changed": changed, "policy_path": self.policy_path}
