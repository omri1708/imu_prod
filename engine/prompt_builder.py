# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, time
from typing import Dict, Any, List

DEFAULT_TEMPLATES = [
  {"id":"plain",   "sys":"פעל כשולחן עבודה הנדסי. ענה תמציתי וענייני.", "usr":"{prompt}"},
  {"id":"planner", "sys":"אתה מתכנן. החזר שלבים/תלויות/בדיקות. אין רטוריקה.", "usr":"משימה: {prompt}"},
  {"id":"json",    "sys":"החזר JSON תקין בלבד. בלי טקסט חופשי.", "usr":"{prompt}\n\nסכמה: {schema_hint}"}
]

class PromptBuilder:
    """
    רישום תבניות + למידה קלה (bandit) לפי הצלחה/כשל/latency.
    שומר סטטיסטיקות לקובץ ומעדכן הסתברויות בחירה.
    """
    def __init__(self, templates_path:str, stats_path:str):
        self.templates_path = templates_path
        self.stats_path = stats_path
        self.templates = self._load_templates()
        self.stats = self._load_stats()   # {template_id: {"n":..,"ok":..,"ms_sum":..}}

    def _load_templates(self) -> List[Dict[str,Any]]:
        if os.path.exists(self.templates_path):
            try: return [json.loads(l) for l in open(self.templates_path,"r",encoding="utf-8")]
            except: pass
        return DEFAULT_TEMPLATES

    def _load_stats(self) -> Dict[str,Any]:
        if os.path.exists(self.stats_path):
            try: return json.loads(open(self.stats_path,"r",encoding="utf-8").read())
            except: pass
        return {t["id"]: {"n":0,"ok":0,"ms_sum":0.0} for t in self.templates}

    def _save_stats(self):
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        open(self.stats_path,"w",encoding="utf-8").write(json.dumps(self.stats, ensure_ascii=False))

    def _pick(self, task:str, intent:str, json_only:bool) -> Dict[str,Any]:
        # בחירה אופטימיסטית (UCB1) פשטנית
        import math, random
        cand = [t for t in self.templates if (not json_only or t["id"]=="json")]
        total = sum(self.stats[t["id"]]["n"] for t in cand) + 1e-9
        def score(t):
            s = self.stats[t["id"]]; n, ok = s["n"], s["ok"]
            mean = (ok/(n or 1))
            ucb = math.sqrt(2*math.log(total)/(n or 1))
            return mean + 0.1*ucb
        cand.sort(key=score, reverse=True)
        return cand[0]

    def compose(self, user_id:str, task:str, intent:str,
                persona:Dict[str,Any], content:Dict[str,Any], json_only:bool=False) -> List[Dict[str,str]]:
        t = self._pick(task, intent, json_only)
        sys = t["sys"] + f"\n[persona]={json.dumps(persona, ensure_ascii=False)}\n[task]={task} [intent]={intent}"
        usr = t["usr"].format(**content)
        return [{"role":"system","content":sys},{"role":"user","content":usr}]

    def learn(self, user_id:str, task:str, intent:str, success:bool, latency_ms:int|None=None, error:str|None=None):
        for t in self.templates:
            tid=t["id"]; s=self.stats.setdefault(tid, {"n":0,"ok":0,"ms_sum":0.0})
            s["n"] += 1
            if success: s["ok"] += 1
            if latency_ms: s["ms_sum"] += latency_ms
        self._save_stats()
