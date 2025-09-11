# engine/prompt_builder.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, math, time, hashlib
from typing import Dict, Any, List


DEFAULT_TEMPLATES = [
  {"id":"plain",   "sys":"פעל כשולחן עבודה הנדסי. ענה תמציתי וענייני.", "usr":"{prompt}"},
  {"id":"planner", "sys":"אתה מתכנן. החזר שלבים/תלויות/בדיקות. אין רטוריקה.", "usr":"משימה: {prompt}"},
  {"id":"json",    "sys":"החזר JSON תקין בלבד. בלי טקסט חופשי.", "usr":"{prompt}\n\nסכמה: {schema_hint}"},
  {"id":"ux_discovery","sys":"אתה אנליסט מוצר. תשאל שאלות קצרות ומעשיות כדי להשלים אפיון UX (מסכים, ניווט, הרשמה, פעולות, נתונים, בידול). בלי ז'רגון טכני.","usr":"בקשה: {prompt}\nאם חסרים פרטים, שאל ברשימת שאלות ממוספרת."},
  {"id":"spec_refine","sys":"אתה מתכנן מערכות. הפוך אפיון UX למסמך JSON קנוני: components, screens, nav, data, flows, differentiation. החזר JSON תקין בלבד.","usr":"תוכן:\n{prompt}\nסכמה: {schema_hint}"},
]

def _stats_skeleton(templates: List[Dict[str,Any]]) -> Dict[str,Any]:
    return {t["id"]: {"n":0,"ok":0,"ms_sum":0.0,"last":0} for t in templates}

class _SafeDict(dict):
    def __missing__(self, key):
        return ""

class PromptBuilder:
    def __init__(self, templates_path:str="./config/prompt_templates.jsonl", stats_path:str="./assurance_store/pb.stats.json"):
        self.templates_path = templates_path
        self.stats_path = stats_path
        self.templates = self._load_templates()
        self.stats = self._load_stats()
        # ודא שלכל תבנית יש רשומת סטטיסטיקה
        for t in self.templates:
            self.stats.setdefault(t["id"], {"n":0,"ok":0,"ms_sum":0.0,"last":0})

    def _load_templates(self) -> List[Dict[str,Any]]:
        if self.templates_path and os.path.exists(self.templates_path):
            try:
                return [json.loads(x) for x in open(self.templates_path,"r",encoding="utf-8")]
            except Exception:
                pass
        return DEFAULT_TEMPLATES

    def _load_stats(self) -> Dict[str,Any]:
        if self.stats_path and os.path.exists(self.stats_path):
            try:
                data = json.loads(open(self.stats_path,"r",encoding="utf-8").read())
                # מסכים שדות חסרים/שבורים
                if not isinstance(data, dict):
                    raise ValueError()
                for k,v in list(data.items()):
                    if not isinstance(v, dict):
                        data[k] = {"n":0,"ok":0,"ms_sum":0.0,"last":0}
                    else:
                        v.setdefault("n",0)
                        v.setdefault("ok",0)
                        v.setdefault("ms_sum",0.0)
                        v.setdefault("last",0)
                return data
            except Exception:
                pass
        return _stats_skeleton(self.templates)

    def _save_stats(self):
        if not self.stats_path:
            return
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        open(self.stats_path,"w",encoding="utf-8").write(json.dumps(self.stats, ensure_ascii=False, indent=2))

    def _pick(self, task:str, intent:str, json_only:bool) -> Dict[str,Any]:
        cand = [t] if (json_only and (t:=self._get_json_template())) else list(self.templates)

        # סינון אם json_only ועדיין אין "json" — נ fallback ל־plain
        if json_only and len(cand)==0:
            cand = [self._get_plain_template()]

        # הגנה: אם אין בכלל תבניות, השתמש ב־DEFAULT_TEMPLATES
        if not cand:
            cand = DEFAULT_TEMPLATES

        # UCB יציב: משתמשים ב-log1p כדי למנוע sqrt שלילי, ומגינים מפני n==0
        total_n = 0
        for t in cand:
            s = self.stats.get(t["id"], {"n":0})
            n = int(s.get("n",0) or 0)
            if n < 0:
                n = 0
            total_n += n

        total_n = max(0, total_n)  # לא שלילי
        log_term = math.log1p(total_n)  # >= 0 אפילו כשהכל אפס

        def score(tpl: Dict[str,Any]) -> float:
            s = self.stats.get(tpl["id"], {"n":0,"ok":0,"last":0})
            n  = int(s.get("n",0) or 0)
            ok = int(s.get("ok",0) or 0)
            # ממוצע אמפירי עם prior עדין כדי לא להפלות תבניות חדשות
            mean = (ok + 0.5) / ((n if n>0 else 0) + 1.0)
            denom = float(n if n>0 else 1)
            ucb = math.sqrt(2.0 * log_term / denom)  # לא שלילי
            # שובר שוויון דטרמיניסטי ע"י hash id
            tie = (int(hashlib.sha256(tpl["id"].encode()).hexdigest(),16) % 997) / 997.0 * 1e-6
            return mean + 0.1*ucb + tie

        cand.sort(key=score, reverse=True)
        return cand[0]

    def _get_json_template(self) -> Dict[str,Any] | None:
        for t in self.templates:
            if t.get("id") == "json":
                return t
        return None

    def _get_plain_template(self) -> Dict[str,Any]:
        for t in self.templates:
            if t.get("id") == "plain":
                return t
        return DEFAULT_TEMPLATES[0]


    def compose(self, user_id: str, task: str, intent: str,
                persona: Dict[str, Any], content: Dict[str, Any], json_only: bool = False) -> List[Dict[str, str]]:
        t = self._pick(task, intent, json_only)

        # אם התבנית דורשת schema_hint והוא לא קיים — fallback ל־plain
        if "{schema_hint}" in t["usr"] and not content.get("schema_hint"):
            t = self._get_plain_template()

        ctx = content.get("context") or {}
        sys = (
            t["sys"]
            + f"\n[persona]={json.dumps(persona, ensure_ascii=False)}"
            + f"\n[context]={json.dumps(ctx, ensure_ascii=False)}"
            + f"\n[task]={task} [intent]={intent}"
        )

        # מפה בטוחה עם דיפולטים ריקים למפתחות חסרים
        mapping = _SafeDict({
            "prompt": str(content.get("prompt", "")),
            "schema_hint": str(content.get("schema_hint", "")),
        })

        usr = t["usr"].format_map(mapping)
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


    def learn(self, template_id:str, user_id:str, task:str, intent:str, success:bool, latency_ms:float|None=None, error:str|None=None):
        # מעדכן את כל התבניות שהיו בשימוש לאחרונה (כאן אין tracking פר תבנית, אז נעדכן כולן בעדינות)
        now_ms = int(time.time()*1000)
        st = self.stats.setdefault(template_id, {"n":0,"ok":0,"ms_sum":0.0})
        st ["n"] += 1
        st["ok"] += 1 if success else 0
        if latency_ms:
            st["ms_sum"] += float(latency_ms)
        for tid, s in self.stats.items():
            if not isinstance(s, dict):
                s = {}
            s["n"] = int(s.get("n",0) or 0) + 1
            if success:
                s["ok"] = int(s.get("ok",0) or 0) + 1
            if latency_ms:
                s["ms_sum"] = float(s.get("ms_sum",0.0) or 0.0) + float(latency_ms)
            s["last"] = now_ms
            self.stats[tid] = s
        self._save_stats()

