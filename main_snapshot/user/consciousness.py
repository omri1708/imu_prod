# imu_repo/user/consciousness.py
from __future__ import annotations
import os, time, hashlib, math
from typing import Dict, Any, Optional, List, Tuple

from user.crypto_store import EncryptedJSONStore, ResourceRequired
from user.memory_state import MemoryState

AFFECT_LABELS = ["calm","curious","frustrated","angry","excited","sad","neutral"]


class UserMind:
    """
    מצב משתמש “חי”: affect (רגש), goals (מטרות), beliefs (אמונות/העדפות), וממשק ללמידה מתמשכת.
    """
    def __init__(self, user_id: str, mem: Optional[MemoryState] = None):
        self.user_id=user_id
        self.mem = mem or MemoryState(user_id)
        self.affect="neutral"; self.aff_conf=0.6
        self.goals: Dict[str,Any] = {}   # {"build_any_app": True, ...}
        self.beliefs: Dict[str,Dict[str,Any]] = {}  # key -> {"value":..., "conf":..., "ts":...}

    # ---- עדכונים ----
    def observe_emotion(self, signal: str, strength: float = 0.7):
        if signal not in AFFECT_LABELS: return
        if strength >= self.aff_conf:
            self.affect=signal; self.aff_conf=strength
        self.mem.add_observation("affect", f"affect:{signal}", {"strength":strength}, conf=strength, ttl_s=7*24*3600, tier="T1")

    def set_goal(self, name: str, value: Any, conf: float = 0.7):
        self.goals[name]=value
        self.mem.add_observation("goal", f"{name}:{value}", {"goal":name}, conf=conf, ttl_s=30*24*3600, tier="T2")

    def assert_belief(self, key: str, value: Any, conf: float = 0.7):
        # איחוד סתירות: אם יש belief קודם — נעדכן לפי confidence+recency
        prev=self.beliefs.get(key)
        ts=time.time()
        take=True
        if prev:
            if prev["conf"]>conf and (ts-prev["ts"])<14*24*3600:
                take=False
        if take:
            self.beliefs[key]={"value":value,"conf":conf,"ts":ts}
        # כתיבה לזיכרון מתמשך
        self.mem.add_observation("belief", f"{key}={value}", {"key":key}, conf=conf, ttl_s=180*24*3600, tier="T2")

    # ---- שאילת זיכרון + התאמת Routing ----
    def recall(self, query: str, k: int = 5) -> List[Dict[str,Any]]:
        res = self.mem.query(query, k)
        # נשמור קוורי אחרון ל-salience
        recent=self.mem.t0.get("__recent_queries__", [])
        recent.append(query); recent=recent[-10:]
        self.mem.t0["__recent_queries__"]=recent
        return res

    def consolidate(self):
        return self.mem.consolidate()

    def decay(self): self.mem.decay()

    # ---- השפעה על Engine ----
    def routing_hints(self) -> Dict[str,Any]:
        """
        מייצר רמזים ל-Engine (policy, limits, סגנון, קישוטי UI) לפי מצב רגשי/מטרות/אמונות.
        דוגמה: אם משתמש “frustrated” → להגביר הסברים ולקצר חיפוש.
        """
        hints={"explain_level":"normal","search_depth":"normal","strict_grounding":True}
        if self.affect in ("frustrated","angry"):
            hints.update({"explain_level":"detailed","search_depth":"shallow"})
        if self.goals.get("build_any_app"):
            hints.update({"search_depth":"deep"})
        return hints


class ConsentError(Exception): ...


class ConflictError(Exception): ...


def _score(rec: Dict[str,Any], now: float, half_life: float = 7*24*3600) -> float:
    """ציון החלטה: שילוב רסנסי + אמון + תמיכה."""
    age = now - rec.get("ts", now)
    recency = math.exp(-age / max(1.0, half_life))
    trust   = float(rec.get("trust", 0.5))
    support = math.log(1.0 + float(rec.get("support", 1.0)))
    return 0.5*recency + 0.4*trust + 0.1*support


class UserConsciousness:
    """
    תודעת משתמש מוצפנת:
    beliefs: {key: {value, ts, ttl, trust, support, sources[]}}
    goals:   {goal: {priority, ts}}
    emotions:{ts: {emotion, intensity}}
    culture: {locale, norms...}
    history: [ {text, embedding[...], ts} ]
    """

    def __init__(self, root: str = ".imu_state/consciousness", password: str = "imu", strict_security: bool = False):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.strict = strict_security
        self.password = password

    def _path(self, user_id: str) -> str:
        h = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        return os.path.join(self.root, f"{h}.encjson")

    def _store(self, user_id: str) -> EncryptedJSONStore:
        return EncryptedJSONStore(self._path(user_id), password=self.password, strict_security=self.strict)

    def _load(self, user_id: str) -> Dict[str, Any]:
        return self._store(user_id).all() or {"beliefs":{}, "goals":{}, "emotions":{}, "culture":{}, "consent":False, "history":[]}

    def _save(self, user_id: str, data: Dict[str, Any]):
        self._store(user_id).put("__root__", data)

    # --- consent ---
    def grant_consent(self, user_id: str):
        d = self._load(user_id); d["consent"] = True; self._save(user_id, d)

    def revoke_consent(self, user_id: str):
        d = self._load(user_id); d["consent"] = False; self._save(user_id, d)

    # --- beliefs ---
    def update_belief(self, user_id: str, key: str, value: Any, trust: float = 0.6,
                      ttl: int = 0, sources: Optional[List[str]] = None):
        d = self._load(user_id)
        if not d.get("consent"): raise ConsentError("consent_required")
        rec = {"value": value, "ts": time.time(), "ttl": ttl, "trust": float(trust),
               "support": 1.0, "sources": sources or []}
        d.setdefault("beliefs", {})
        prior = d["beliefs"].get(key)
        if prior:
            # אם אותו הערך – לחזק תמיכה; אחרת נוצרה סתירה
            if prior.get("value") == value:
                rec["support"] = float(prior.get("support",1.0)) + 1.0
                rec["trust"] = max(float(prior.get("trust",0.0)), rec["trust"])
        d["beliefs"][key] = rec
        self._save(user_id, d)

    def recall_belief(self, user_id: str, key: str) -> Optional[Any]:
        d = self._load(user_id)
        rec = d.get("beliefs",{}).get(key)
        if not rec: return None
        ttl = rec.get("ttl",0)
        if ttl and time.time() - rec.get("ts",0) > ttl:
            del d["beliefs"][key]; self._save(user_id, d); return None
        return rec.get("value")

    def resolve_conflicts(self, user_id: str):
        """בחירה ערכית לפי ציון (recency+trust+support)."""
        d = self._load(user_id)
        now = time.time()
        beliefs = d.get("beliefs", {})
        # כאן הדגם: אם יש מפתחות עם ערכי מועמדים שונים (נדרש ייצוג מרובה-גרסאות)
        # נייצג כ-beliefs_variants: key -> [rec1, rec2...]
        variants = d.get("beliefs_variants", {})
        for key, recs in list(variants.items()):
            best = max(recs, key=lambda r: _score(r, now))
            beliefs[key] = best
        d["beliefs"] = beliefs
        d["beliefs_variants"] = {}
        self._save(user_id, d)

    # --- emotions, goals, culture ---
    def record_emotion(self, user_id: str, emotion: str, intensity: float):
        d = self._load(user_id)
        if not d.get("consent"): raise ConsentError("consent_required")
        d.setdefault("emotions", {})
        d["emotions"][str(time.time())] = {"emotion":emotion, "intensity":float(intensity)}
        self._save(user_id, d)

    def add_goal(self, user_id: str, goal: str, priority: int = 1):
        d = self._load(user_id)
        if not d.get("consent"): raise ConsentError("consent_required")
        d.setdefault("goals", {})
        d["goals"][goal] = {"priority": int(priority), "ts": time.time()}
        self._save(user_id, d)

    def cultural_context(self, user_id: str, context: Dict[str, Any]):
        d = self._load(user_id)
        if not d.get("consent"): raise ConsentError("consent_required")
        d.setdefault("culture", {})
        d["culture"].update(context)
        self._save(user_id, d)

    # --- semantic history ---
    def _hash_embedding(self, text: str, dim: int = 64) -> List[float]:
        # Embedding דטרמיניסטי קל משקל: חלוקה ל-chunks והאש לכל bucket
        vec = [0.0]*dim
        words = text.split()
        for i, w in enumerate(words):
            h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        # נרמול
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x/norm for x in vec]

    def semantic_learn(self, user_id: str, text: str, embedding: Optional[List[float]] = None):
        d = self._load(user_id)
        if not d.get("consent"): raise ConsentError("consent_required")
        emb = embedding or self._hash_embedding(text)
        d.setdefault("history", []).append({"text": text, "embedding": emb, "ts": time.time()})
        self._save(user_id, d)
