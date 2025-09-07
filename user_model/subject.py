# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Optional
import re, time
from user_model.model import UserModel, UserStore, Preference

class SubjectEngine:
    """
    שכבה דקה מעל UserModel: מזהה דפוסים ומעדכן פרופיל הפרט (העדפות/מטרות) עם confidence.
    """
    def __init__(self, store: UserStore):
        self.um = UserModel(store)

    def observe_text(self, uid: str, text: str):
        """
        ניתוח פשוט: נזהה מילות מפתח (dark/light, lang:he/en, perf/safety וכו') ונעדכן העדפות.
        זה הדגמה; בפועל מחליפים במודלי embedding/sequence.
        """
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

    def subject_profile(self, uid: str) -> Dict[str,Any]:
        """ייצוא פרופיל: הערכים האחרונים בעלי confidence הגבוה."""
        prof={}
        for key in ("ui.theme","ui.lang","routing.bias"):
            p = self.um.pref_get(uid, key)
            if p: prof[key]= {"value": p.value, "confidence": p.confidence, "ts": p.ts}
        return prof
