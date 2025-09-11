# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, time, hashlib
from typing import Dict, Any
from engine.llm_gateway import LLMGateway
from engine.privacy.memory_policy import apply_ttl, scope_filter, consent_ok

ROOT = "./assurance_store_users"
os.makedirs(ROOT, exist_ok=True)

T0_MAX_MSGS   = 50          # חלון שיחה קצר
T1_MAX_CHUNKS = 20          # כמה סיכומי־עבר נשמור
T2_MAX_FACTS  = 200         # כמה עובדות “קשיחות” נשמור
TTL_DAYS_T1   = 180
TTL_DAYS_T2   = 365

SECRET_PAT = re.compile(r"(?i)\b(api[_-]?key|token|password|secret|bearer)\b[:= ]+[A-Za-z0-9\-\._~]+")
EMAIL_PAT  = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

def _now_ms() -> int: return int(time.time()*1000)
def _days(ms:int) -> float: return ms / (1000*60*60*24)

def mem_path(uid:str) -> str:
    uid_s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", uid or "user")
    return os.path.join(ROOT, f"{uid_s}.mem.json")

def _load(uid:str) -> Dict[str,Any]:
    p = mem_path(uid)
    if not os.path.exists(p):
        return {"uid":uid,"t0":[],"t1":[],"t2":[],"ts":_now_ms()}
    try:
        return json.loads(open(p,"r",encoding="utf-8").read())
    except Exception:
        return {"uid":uid,"t0":[],"t1":[],"t2":[],"ts":_now_ms()}

def _save(uid:str, st:Dict[str,Any]):
    open(mem_path(uid),"w",encoding="utf-8").write(json.dumps(st, ensure_ascii=False))

def _redact(txt:str) -> str:
    txt = SECRET_PAT.sub("[REDACTED]", txt)
    txt = EMAIL_PAT.sub("[email]", txt)
    return txt

def _hash(s:str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

class MemoryBridge:
    """
    T0: חלון הודעות קצר (שני הכיוונים).
    T1: סיכומים אפיזודיים + עובדות (facts) נמשכות מהשיחה.
    T2: עובדות סמנטיות “קשיחות” (דחוסות/מאוחדות), עם TTL.
    הכל עובר רידקציה/TTL ו-Consolidation עם LLMGateway (JSON-only), ונשמר כמצב משתמש.
    """
    def __init__(self):
        self.gw = LLMGateway()


    # observe_turn: קודם רידקציה, ואז כתיבה
    def observe_turn(self, uid:str, role:str, text:str):
        st = _load(uid)
        text = _redact(text)  # ← להזיז למעלה
        st["t0"].append({
            "ts": _now_ms(), "role": role, "text": text,
            "scope":"t0", "ttl_s": 7*24*3600, "consent": True
        })
        if len(st["t0"]) > T0_MAX_MSGS:
            self._consolidate_t0_to_t1(uid, st)
            st["t0"] = st["t0"][-(T0_MAX_MSGS//2):]
        self._gc(st)
        st["ts"] = _now_ms()
        _save(uid, st)

    # pack_context: להשתמש ב-scope_filter
    def pack_context(self, uid:str, user_query:str) -> Dict[str,Any]:
        st = _load(uid)
        recent = st["t0"][-8:]

        episodic_all = st.get("t1") or []
        episodic = [x for x in episodic_all if apply_ttl(x) and consent_ok(x) and scope_filter(x, {"t1"})][-3:]

        q = (user_query or "").lower()
        toks = set(re.findall(r"[A-Za-zא-ת0-9]{3,}", q))
        def rel_score(fact:str)->int: return sum(1 for tok in toks if tok.lower() in fact.lower())
        facts_all = [f for f in (st.get("t2") or []) if apply_ttl(f) and consent_ok(f) and scope_filter(f, {"t2"})]
        facts_sorted = sorted(facts_all, key=lambda f: rel_score(f.get("fact","")), reverse=True)[:10]

        return {"t0_recent": recent, "t1_episodic": episodic, "t2_facts": facts_sorted}


    # Summarization (T0 -> T1)
    def _consolidate_t0_to_t1(self, uid:str, st:Dict[str,Any]):
        old_half = st["t0"][:len(st["t0"])//2]
        if not old_half:
            return
        text_blob = "\n".join(f"{m['role']}: {m['text']}" for m in old_half)
        schema = '{"summary":"string","facts":["string"]}'
        res = self.gw.structured(user_id=uid, task="memory", intent="summarize",
                                 schema_hint=schema,
                                 prompt=f"סכם את ההודעות הבאות ותמצת עובדות יציבות לפרסונה. החזר JSON בלבד.\n{text_blob}",
                                 temperature=0.0)
        j = res["json"]
        item = {"ts": _now_ms(), "summary": _redact(j.get("summary","")),
                "facts": [], "scope":"t1", "ttl_s": TTL_DAYS_T1*24*3600, "consent": True}
        for f in j.get("facts") or []:
            f = _redact(f).strip()
            if not f:
                continue
            h = _hash(f)
            if not any(x.get("hash")==h for x in (st.get("t2") or [])):
                (st["t2"]).append({
                    "ts": _now_ms(), "hash": h, "fact": f, "confidence": 0.7,
                    "scope":"t2", "ttl_s": TTL_DAYS_T2*24*3600, "consent": True
                })

            item["facts"].append(f)
        (st["t1"]).append(item)
        if len(st["t1"]) > T1_MAX_CHUNKS:
            st["t1"] = st["t1"][-T1_MAX_CHUNKS:]
        if len(st.get("t2",[])) > T2_MAX_FACTS:
            st["t2"] = st["t2"][-T2_MAX_FACTS:]

    # ניקוי TTL
    def _gc(self, st:Dict[str,Any]):
        now = _now_ms()
        st["t1"] = [x for x in (st.get("t1") or []) if _days(now - x.get("ts", now)) <= TTL_DAYS_T1]
        st["t2"] = [x for x in (st.get("t2") or []) if _days(now - x.get("ts", now)) <= TTL_DAYS_T2]

    # איפוס T0 בלבד (מבלי למחוק T1/T2)
    def reset_t0(self, uid:str):
        st = _load(uid)
        st["t0"] = []
        _save(uid, st)

    def wipe_user(self, uid: str):
        try:
            os.remove(mem_path(uid))
        except Exception:
            pass

# מופע יחיד לשימוש בצ'אט
MB = MemoryBridge()