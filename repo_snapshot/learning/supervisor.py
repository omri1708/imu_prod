# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio, time, os, json
from typing import List
from learning.event_sink import iter_audit_events
from learning.policy_learner import PolicyTuner, MetricsSnapshot

class LearningSupervisor:
    def __init__(self, policy_path: str, adapters_root: str,
                 audit_roots: List[str], rr_log_path: str,
                 period_seconds: int = 30):
        self.tuner = PolicyTuner(policy_path, adapters_root)
        self.audit_roots = audit_roots
        self.rr_log_path = rr_log_path
        self.period = period_seconds

    def _drain_resource_required_log(self):
        """
        קורא ./logs/resource_required.jsonl (אם קיים) ומעדכן את ה-Tuner.
        פורמט השורה: {"ts":..., "what":"tool:curl"|"cmd_not_found:gcc"|...}
        """
        p = self.rr_log_path
        if not os.path.exists(p): 
            return
        new_lines=[]
        with open(p,"r+",encoding="utf-8") as f:
            for line in f:
                try:
                    j=json.loads(line)
                    w=j.get("what")
                    if w:
                        self.tuner.observe_resource_required(w)
                except Exception:
                    pass
            # ננקה את הקובץ לאחר קריאה (פשטות)
            f.seek(0); f.truncate()

    async def run_forever(self):
        while True:
            try:
                # איסוף audit (כרגע לא מנצל counters לפני-commit; די לקלוט commits)
                events = list(iter_audit_events([os.path.join(r, "") for r in self.audit_roots if os.path.exists(r)]))
                m = MetricsSnapshot(total_commits=len(events), resource_required_counts=None)
                # ריקון לוג משאבים חסרים
                self._drain_resource_required_log()
                # עדכון מדיניות (כללים פשוטים)
                self.tuner.tune_from_metrics(m)
            except Exception as e:
                # לא מפילים את השרת; נמשיך בלולאה
                pass
            await asyncio.sleep(self.period)
