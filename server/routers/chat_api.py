# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import json
from typing import Dict, Any, Optional, Tuple, List

from fastapi import APIRouter, HTTPException, Query

# -------- Imports (as in your project layout) --------
from server.dialog.state import SessionState
from server.dialog.memory_bridge import MB

from engine.llm_gateway import LLMGateway
from engine.intent_to_spec import IntentToSpec
from engine.spec_refiner import SpecRefiner
from engine.build_orchestrator import BuildOrchestrator
from engine.orchestrator.universal_orchestrator import (
    UniversalOrchestrator,
    OrchestratorConfig,   # מאפשר בחירה בין poc/live/prod
)

from user_model.model import UserStore
from user_model.subject import SubjectEngine
from grounded.fact_gate import RefusedNotGrounded

# =====================================================
#  Chat API Router — Conversation + Grounding + Build
# =====================================================

router = APIRouter(prefix="/chat", tags=["chat"])

# --- Global runtime singletons (שימוש־חוזר) ---
SESS: Dict[str, SessionState] = {}
STORE = UserStore("./assurance_store_users")
SUBJECT = SubjectEngine(STORE)

# Gateway יחיד לכל ה־router (כולל פרסונה/סטטיסטיקות)
GW = LLMGateway(user_store_path="./assurance_store_users")

# תומכים גם במסלול בנייה ישיר (כשצריך)
BUILDER = BuildOrchestrator()

# Orchestrators לפי מצב עבודה (poc / live / prod) — מטמון
_ORCS: Dict[str, UniversalOrchestrator] = {}

# ------------------------- Utilities -------------------------

def _say(uid: str, st: SessionState, text: str, extra: Optional[Dict[str, Any]] = None):
    """רישום תשובת assistant בזיכרון + החזרת JSON עקבי לקליינט."""
    MB.observe_turn(uid, "assistant", text)
    st.last_text = text  # type: ignore[attr-defined]
    return {"ok": True, "text": text, "extra": extra or {}}

def _extract_urls(text: str) -> List[str]:
    return re.findall(r"https?://\S+", text or "")

def _sanitize_prompt(text: str) -> str:
    return re.sub(r"https?://\S+", "", text).strip()

def _persona_for_context(uid: str) -> Dict[str, Any]:
    """פרסונה/העדפות לשילוב בקונטקסט LLM."""
    # נלמד העדפות מהטקסט; נשלוף גם פרופיל ופרסונה
    prof = SUBJECT.subject_profile(uid)  # ע"פ העדפות יציבות
    persona = SUBJECT.persona(uid)       # מיזוג פרסונה+העדפות עדכניות
    return {"profile": prof, "persona": persona}

def _orc_for_mode(mode: str, auto_install: Optional[bool] = None) -> UniversalOrchestrator:
    """החזרת UniversalOrchestrator לפי מצב עבודה, עם קונפיג מתאים."""
    key = f"{mode}:{'auto' if (auto_install is None) else str(bool(auto_install))}"
    if key not in _ORCS:
        cfg = OrchestratorConfig(mode=mode, auto_install=auto_install)
        _ORCS[key] = UniversalOrchestrator(gateway=GW, config=cfg)
    return _ORCS[key]

def _merge_specs(primary: Dict[str, Any], aux: Dict[str, Any]) -> Dict[str, Any]:
    """
    מיזוג עדין: primary הוא spec ראשי (מ־UniversalPlanner),
    aux (למשל IntentToSpec/Refiner) מוסיף ישויות/אירועים/כללים/מדדים.
    """
    out = dict(primary)
    def _merge_list(key: str):
        a = list(out.get(key) or [])
        b = list(aux.get(key) or [])
        seen = set(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in a)
        for x in b:
            sx = json.dumps(x, ensure_ascii=False, sort_keys=True)
            if sx not in seen:
                a.append(x)
                seen.add(sx)
        out[key] = a
    for k in ("entities", "events", "rules", "features", "success_metrics", "acceptance_criteria"):
        _merge_list(k)
    # שמור מקור
    out.setdefault("__provenance__", {})
    out["__provenance__"].update({"intent_to_spec": bool(aux)})
    return out

def _planning_assist(uid: str, spec: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    עוזר תכנון מובנה (JSON בלבד): דרישות, אילוצים, trade-offs, חלופות ארכיטקטורה,
    IaC/CI-CD, ניטור, סיכונים ושאלות פתוחות. מבוסס LLMGateway.structured.
    """
    schema = json.dumps({
        "requirements": {"explicit": ["string"], "implicit": ["string"], "constraints": ["string"]},
        "tradeoffs": [{"option": "string", "pros": ["string"], "cons": ["string"], "when": ["string"], "risks": ["string"]}],
        "architecture": {"styles": ["string"], "preferred": "string", "why": "string", "microservices": True},
        "infra": {"cloud": "string", "iac": "terraform|pulumi|cdk|other", "services": ["string"], "observability": ["string"]},
        "cicd": {"system": "github_actions|gitlab|circle|argo|other", "stages": ["string"], "environments": ["string"]},
        "monitoring": {"metrics": ["string"], "alerts": ["string"], "slos": ["string"]},
        "open_questions": ["string"],
        "unknowns": ["string"]
    }, ensure_ascii=False)
    prompt = (
        "Analyze the following SPEC and user context. Return JSON ONLY, per schema.\n"
        "Be concrete and implementation-ready (no templates). Keep lists concise but complete.\n"
        f"SPEC:\n{json.dumps(spec, ensure_ascii=False)}\n"
        f"CONTEXT:\n{json.dumps(ctx, ensure_ascii=False)}"
    )
    res = GW.structured(
        user_id=uid, task="planning", intent="build_architecture",
        schema_hint=schema, prompt=prompt, temperature=0.1
    )
    return res.get("json") or {}

# -------------------------- Public Endpoints --------------------------

@router.get("/history")
def history(user_id: str = Query(...), limit: int = Query(40, ge=1, le=200)):
    ctx = MB.pack_context(user_id, "")
    t0 = ctx.get("t0_recent") or []
    return {"ok": True, "history": t0[-limit:]}

@router.post("/reset")
def reset(body: Dict[str, Any]):
    uid = (body.get("user_id") or "user").strip()
    MB.reset_t0(uid)            # שומר T1/T2 (זיכרון ארוך טווח), מאפס חלון שיחה
    SESS.pop(uid, None)
    return {"ok": True}

@router.post("/send")
async def send(body: Dict[str, Any]):
    """
    מסלול שיחה חכם:
      1) Grounded QA (sources או URLs) → תשובה ללא הלוצינציות + ציטוטים
      2) אחרת: Discovery→Spec (UniversalPlanner) → Refine→Analysis → Execute (Build)
         כולל שאלות פתוחות, דרישות נסתרות, trade-offs ו־CI/CD/IaC לפי ההקשר
    """
    uid = (body.get("user_id") or "user").strip()
    msg = (body.get("message") or "").strip()
    if not msg:
        raise HTTPException(400, "message required")

    # מצב/קונטקסט שיחה
    st = SESS.setdefault(uid, SessionState())
    MB.observe_turn(uid, "user", msg)
    SUBJECT.observe_text(uid, msg)

    # ריכוז קונטקסט מרלוונטי (T0/T1/T2) + פרסונה
    ctx = MB.pack_context(uid, msg)
    ctx.update(_persona_for_context(uid))

    # בחירת מצב Orchestrator לפי הבקשה (או ברירת מחדל 'live')
    mode = (body.get("mode") or "").strip().lower() or "live"
    auto_install = body.get("auto_install", None)  # None ⇒ לפי ברירות מחדל של המוד
    ORC = _orc_for_mode(mode, auto_install=auto_install)

    # ======================================
    # A) Strict grounded answer (no hallucinations)
    # ======================================
    sources: List[str] = body.get("sources") or _extract_urls(msg)
    grounded_flag = bool(body.get("grounded") or msg.startswith("ענה"))

    if grounded_flag or sources:
        prompt = _sanitize_prompt(msg) or "ענה בקצרה"
        out = GW.chat(
            user_id=uid, task="answer", intent="answer",
            content={"prompt": prompt, "sources": sources, "context": ctx},
            require_grounding=True, temperature=float(body.get("temperature", 0.0)),
        )
        if not out.get("ok"):
            # החזרה שקופה של סיבת הדחייה (לא ממחולל)
            detail = out.get("error") or "not_grounded"
            raise HTTPException(400, f"grounded-answer-refused: {detail}")
        payload = out["payload"]
        return _say(uid, st, payload["text"], extra={
            "citations": payload.get("citations"),
            "root": payload.get("root"),
            "grounded": True
        })

    # ======================================
    # B) Build path: Discovery → Spec → Refine → Analysis → Execute
    # ======================================

    # 1) ניתוח בקשה ראשוני (UniversalPlanner בתוך ה־Orchestrator)
    spec_planner = ORC.analyze(uid, msg, ctx)

    # 2) fallback: IntentToSpec (אם המפרט דל מאוד/דומיין לא מזוהה)
    i2s = IntentToSpec()
    try:
        spec_i2s = i2s.from_text(uid, msg)
    except Exception:
        spec_i2s = {}

    spec_merged = _merge_specs(spec_planner, spec_i2s)

    # 3) שיפור/השלמת SPEC לדומיין (entities/behavior/tests) — JSON-only
    refiner = SpecRefiner()
    spec_refined = refiner.refine_if_needed(uid, spec_merged)

    # 4) חקירת דרישות/אילוצים/חלופות ו־trade-offs (Structured JSON)
    analysis = _planning_assist(uid, spec_refined, ctx)

    # אם קיימות שאלות פתוחות והלקוח לא ביקש "proceed" — נעצור לשאלות ממוקדות
    open_q: List[str] = (spec_refined.get("open_questions") or []) + (analysis.get("open_questions") or [])
    proceed = bool(body.get("proceed") or body.get("autobuild"))
    if open_q and not proceed:
        st.stage = "clarify"
        text = "כדי להתקדם לבנייה, נדרש מענה לכמה שאלות ממוקדות."
        return _say(uid, st, text, extra={
            "mode": mode,
            "next": {"type": "questions", "items": open_q[:10]},
            "spec": spec_refined,
            "analysis": analysis,
        })

    # 5) Execute: בנייה אמיתית (כולל lint/compile/pytest אם יש)
    res = await ORC.execute(spec_refined)

    # 6) ניסוח מענה ידידותי למשתמש + החזרת מידע עשיר
    if res.get("ok"):
        txt = "בנוי ✔️"
    elif res.get("still_missing"):
        txt = "נדרשים כלים חיצוניים (ראה extra.instructions)"
    else:
        txt = "הבנייה נכשלה — בדוק לוגים"

    extra = {
        "mode": mode,
        "spec": spec_refined,
        "analysis": analysis,
        **res  # build, tools, missing, instructions, blueprint, domain...
    }
    return _say(uid, st, txt, extra=extra)
