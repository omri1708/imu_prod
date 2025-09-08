# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Optional
import re, time
from user_model.model import UserModel, UserStore, Preference
# -*- coding: utf-8 -*-


class SubjectEngine:
    """
    שכבה דקה מעל UserModel: מזהה דפוסים ומעדכן פרופיל הפרט (העדפות/מטרות) עם confidence.
    """
    def __init__(self, store: UserStore) -> None:
        self.um = UserModel(store)
        self.store = store

    def observe_text(self, uid: str, text: str):
        """
        ניתוח פשוט: נזהה מילות מפתח (dark/light, lang:he/en, perf/safety וכו') ונעדכן העדפות.
        זה הדגמה; בפועל מחליפים במודלי embedding/sequence.
        """
        st = self.store.get(uid)
        prefs = st.get("prefs") or {}
        t = text.lower()
        prefs: List[Preference] = []
        if re.search(r"\bdark\s*mode\b", t):
            self.um.pref_set(uid, "ui.theme", "dark", confidence=0.75)
        if re.search(r"\blight\s*mode\b", t):
            self.um.pref_set(uid, "ui.theme", "light", confidence=0.75)
        if re.search(r"\bhebrew|עברית\b", t):
            self.um.pref_set(uid, "ui.lang", "he", confidence=0.8)
        if re.search(r"\benglish|en-us\b", t):
            self.um.pref_set(uid, "ui.lang", "en", confidence=0.8)
        if re.search(r"\bperformance\b", t):
            self.um.pref_set(uid, "routing.bias", "performance", confidence=0.6)
        if re.search(r"\bsafety\b", t):
            self.um.pref_set(uid, "routing.bias", "safety", confidence=0.6)
        if re.search(r"\b(כהה|דארק|dark)\b", text, re.I):
            prefs["theme"] = "dark"
        if re.search(r"\b(עברית|Hebrew)\b", text, re.I):
            prefs["locale"] = "he-IL"
        if re.search(r"\b(אנדרואיד|android)\b", text, re.I):
            prefs["target_mobile"] = "android"
        if re.search(r"\b(iOS|אייפון|אפל)\b", text, re.I):
            prefs["target_mobile"] = "ios"
        
        st["prefs"] = prefs
        st["last_observe_ts"] = int(time.time()*1000)
        self.store.set(uid, st)

    def subject_profile(self, uid: str) -> Dict[str,Any]:
        """ייצוא פרופיל: הערכים האחרונים בעלי confidence הגבוה."""
        prof={}
        for key in ("ui.theme","ui.lang","routing.bias"):
            p = self.um.pref_get(uid, key)
            if p: prof[key]= {"value": p.value, "confidence": p.confidence, "ts": p.ts}
        return prof

    def persona(self, uid: str) -> Dict[str, Any]:
        st = self.store.get(uid)
        persona = st.get("persona") or {}
        prefs   = st.get("prefs")   or {}
        # מיזוג: פרסונה "קשיחה" + העדפות מהשיחה האחרונה
        merged = {
            "uid": uid,
            "theme": prefs.get("theme") or persona.get("theme") or "light",
            "locale": prefs.get("locale") or persona.get("locale") or "he-IL",
            "targets": {
                "mobile": prefs.get("target_mobile") or persona.get("targets", {}).get("mobile"),
            },
            "product_style": persona.get("product_style") or "guided",  # 'guided'/'power'/'casual'
        }
        return merged
