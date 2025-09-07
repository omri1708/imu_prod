# -*- coding: utf-8 -*-
from __future__ import annotations
import json, asyncio, re
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from server.dialog.intent import detect
from server.dialog.state import SessionState
from server.dialog.planner import normalize_build_request, llm_build_spec

from user_model.model import UserStore, UserModel
from user_model.subject import SubjectEngine
from assurance.respond_text import GroundedResponder
from program.orchestrator import ProgramOrchestrator
from integration.adapter_wrap import run_adapter_with_assurance
from assurance.errors import ResourceRequired, RefusedNotGrounded, ValidationFailed

router = APIRouter(prefix="/chat", tags=["chat"])
SESS: Dict[str,SessionState] = {}
store=UserStore("./assurance_store_users"); subject=SubjectEngine(store)
gr=GroundedResponder("./assurance_store_text"); orch=ProgramOrchestrator("./assurance_store_programs")

def _reply(text:str, extra:Dict[str,Any]|None=None): return {"ok":True,"text":text,"extra":extra or {}}

@router.post("/send")
async def send(payload: Dict[str,Any]):
    uid=(payload.get("user_id") or "user").strip()
    msg=(payload.get("message") or "").strip()
    st=SESS.setdefault(uid, SessionState())
    subject.observe_text(uid, msg)

    # אם יש שאלה פתוחה ונראה שהמשתמש ענה עליה — תמלא ונמשיך
    if st.pending_q:
        low=msg.lower()
        if re.search(r"\b(כן|yes|yep|go|אשר|מאשר)\b", low):
            st.pending_q=None
        else:
            # נסה להשלים לפי תשובה מילולית (למשל סוג)
            if "אתר" in msg or "web" in low: st.slots["type"]="web"; st.pending_q=None
            elif "כלי" in msg or "cli" in low: st.slots["type"]="cli"; st.pending_q=None
            elif "mobile" in low or "מובייל" in msg: st.slots["type"]="mobile"; st.pending_q=None
            elif "desktop" in low or "דסקטופ" in msg: st.slots["type"]="desktop"; st.pending_q=None

    intent, slots = detect(msg)

    # 1) הסכמה טבעית
    if intent=="consent":
        um=UserModel(store); um.identity_register(uid,{"via":"chat"}); um.consent_grant(uid,"adapters/run",slots.get("ttl",86400))
        return _reply("קיבלתי. מעכשיו אוכל להריץ פעולות עבורך.")

    # 2) תשובה ממוסדת (אם לא צוין מקור — נבקש בשאלה אחת)
    if intent=="ask_info":
        urls,files=slots.get("urls",[]), slots.get("files",[])
        if not (urls or files):
            st.pending_q="source"
            return _reply("כדי לענות בלי הלוצינציות אני צריך מקור (URL/קובץ). שלח/י לינק, או כתוב/כתבי 'ביטול'.")
        try:
            out=gr.respond_from_sources(prompt=slots.get("prompt",""), sources=[{"url":u} for u in urls]+[{"file":f} for f in files])
            return _reply(out["payload"]["text"], extra={"citations":out["payload"]["citations"]})
        except ResourceRequired as e:
            return {"ok":False,"text":"נדרש משאב חסר לקבלת המידע.", "extra":{"resource_required":e.what,"obtain":e.how_to_get}}
        except (RefusedNotGrounded,ValidationFailed) as e:
            raise HTTPException(400, f"סירוב: {e}")

    # 3) בניית תוכנית — אתה מדבר, המערכת משלימה שאלות לבד
    if intent=="build_app":
        # ננסה לבנות Spec מלא בעזרת LLM לפי הפרופיל המצטבר; אם LLM כבוי – נשאר עם ההיוריסטיקה הקיימת
        prof = subject.subject_profile(uid)
        spec = llm_build_spec(msg, prof)
        um=UserModel(store)
        if not um.has_consent(uid,"adapters/run"):
            st.pending_q="confirm_build"
            st.slots.update({"name": spec.get("name","app"),
                             "stack": "python_web" if any(s.get("type")=="python_web" for s in spec.get("services",[])) else "python_app"})
            return _reply("אני צריך אישור חד-פעמי להריץ בנייה. כתוב/כתבי: “אני מאשר”.")
        try:
            r=await orch.build(uid, spec)
            st.stage="idle"; st.slots={}; st.pending_q=None
            return _reply("בנוי ✔️", extra=r)
        except ResourceRequired as e:
            return {"ok":False,"text":"חסר משאב כדי להשלים בנייה.", "extra":{"resource_required":e.what,"obtain":e.how_to_get}}
        except ValidationFailed as e:
            raise HTTPException(400, f"שגיאת בדיקות: {e}")

    # אם המשתמש ענה "מאשר" אחרי השאלה
    if st.pending_q=="confirm_build":
        um=UserModel(store)
        if not um.has_consent(uid,"adapters/run"):
            return _reply("אני צריך אישור חד־פעמי כדי להריץ. כתוב/כתבי: “אני מאשר”.")
        try:
            r=await orch.build(uid, {"name":st.slots.get("name","app"),
                                     "services":[{"type": "python_web" if st.slots.get("stack")=="python_web" else "python_app",
                                                  "name": st.slots.get("name","app")}]})
            st.stage="idle"; st.slots={}; st.pending_q=None
            return _reply("בנוי ✔️", extra=r)
        except ResourceRequired as e:
            return {"ok":False,"text":"חסר משאב כדי להשלים בנייה.", "extra":{"resource_required":e.what,"obtain":e.how_to_get}}
        except ValidationFailed as e:
            raise HTTPException(400, f"שגיאת בדיקות: {e}")

    # 4) הפעלת פעולה — גם אם לא ציינת kind במפורש, ננסה להבין או נשאל
    if intent=="run_action":
        if "echo" in msg.lower(): kind="tool.echo"
        else: kind=slots.get("kind")
        if not kind:
            return _reply("איזו פעולה להריץ? לדוגמה: “תריץ echo” או “תריץ tool.echo”.")
        try:
            out=await run_adapter_with_assurance(uid, kind, slots.get("params") or {}, execute=True)
            return _reply("ההרצה הושלמה ✔️", extra=out)
        except ResourceRequired as e:
            return {"ok":False,"text":"חסר משאב כדי להריץ.", "extra":{"resource_required":e.what,"obtain":e.how_to_get}}
        except (ValidationFailed,RefusedNotGrounded) as e:
            raise HTTPException(400, str(e))

    # 5) העדפות (תודעת משתמש) — אין טכני, רק לומדים ומשיבים
    if intent=="preference":
        return _reply("קיבלתי. התאמות נשמרו לפרופיל שלך.", extra={"profile": subject.subject_profile(uid)})

    # 6) ברירת מחדל – עזרה קצרה
    return _reply("ספר לי מה תרצה לבנות או לשאול. למשל: “אני רוצה אפליקציה לניהול משימות”, או “למה השירות איטי? הנה קישור…”.")
