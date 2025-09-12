# server/routers/chat_api.py
from __future__ import annotations
import re
import os 
import json
from typing import Dict, Any, Optional, List
import traceback
from fastapi import APIRouter, HTTPException, Query, Depends, Body, Form
from fastapi.responses import JSONResponse
from engine.diagnostics.traceback_analyzer import parse_traceback
from engine.self_heal.auto_fix import apply_actions
from engine.orchestrator import universal_planner as up
from engine.spec_refiner import SpecRefiner
from server.dialog.state import SessionState
from server.dialog.memory_bridge import MB
from server.deps.evidence_gate import require_citations_or_silence
from engine.orchestrator.run_store import run_context
from server.dialog.intent_router import classify_intent
from engine.orchestrator.universal_orchestrator import UniversalOrchestrator

from engine.llm_gateway import LLMGateway
from engine.intent_to_spec import IntentToSpec
from engine.build_orchestrator import BuildOrchestrator
from engine.orchestrator.universal_orchestrator import (
    UniversalOrchestrator,
    OrchestratorConfig,   # מאפשר בחירה בין poc/live/prod
)
from server.dialog.intent_router import classify_intent, extract_slots
from server.dialog.design_flow import design_arch
from user_model.model import UserStore, UserModel
from user_model.subject import SubjectEngine
from engine.runtime.approvals import required_approvals
from engine.blueprints.registry import resolve as resolve_blueprint
from pathlib import Path
from engine.runtime.consent_store import ConsentStore

CONS = ConsentStore(".imu/consents.json")

# =====================================================
#  Chat API Router — Conversation + Grounding + Build
# =====================================================

router = APIRouter(
     prefix="/chat",
     tags=["chat"],
 )

US = UserStore("./assurance_store_users")
UM = UserModel(US)
UNIV = UniversalOrchestrator()

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

@router.get("/diagnostics")
def diagnostics(user_id: str = "user"):
    """Return LLM readiness for clear root-cause."""
    return {"ok": True, "llm": GW.diagnose()}
# ------------------------- Utilities -------------------------

def _auto_assets(uid: str, spec: Dict[str,Any], ctx: Dict[str,Any], user_msg: str) -> List[Dict[str,Any]]:
    """
    מבקש מה-LLM להפיק assets (קבצים) מתוך הכוונה – ללא מקורות, JSON בלבד:
    [{"path":"ui/chat.html","mime":"text/html","body":"..."}]
    """
    schema = json.dumps({"assets":[{"path":"string","mime":"string","body":"string"}]}, ensure_ascii=False)
    prompt = (
        "From the SPEC and user request, synthesize concrete files to implement the intent. "
        "Return JSON with 'assets' only. Do NOT require external sources.\n"
        f"USER:\n{user_msg}\nSPEC:\n{json.dumps(spec, ensure_ascii=False)}\nCONTEXT:\n{json.dumps(ctx, ensure_ascii=False)}\n"
        "Rules: choose safe relative paths under the project (e.g., ui/*.html, services/api/adapters/*.py, docs/*.md). "
        "Do not include '..' or absolute paths."
    )
    r = GW.structured(uid, task="render", intent="render_assets", schema_hint=schema, prompt=prompt,
                      temperature=0.0, require_grounding=False)
    data = r.get("json") or {}
    assets = data.get("assets") or []
    # סינון בטיחות: בלי יציאה מהעץ
    good = []
    for a in assets:
        p = str(a.get("path") or "")
        b = (a.get("body") or "")
        if not p or not b: 
            continue
        if ".." in p or p.startswith(("/", "\\")): 
            continue
        good.append({"path":p, "mime":str(a.get("mime") or ""), "body":b})
    return good

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
    ignore_memory  = bool(body.get("ignore_memory", False))
    ignore_persona = bool(body.get("ignore_persona", False))

    if ignore_memory:
        ctx = {"t0_recent": [], "t1_episodic": [], "t2_facts": []}
    else:
        ctx = MB.pack_context(uid, msg)

    if not ignore_persona:
        ctx.update(_persona_for_context(uid))

    # בחירת מצב Orchestrator לפי הבקשה (או ברירת מחדל 'live')
    mode = (body.get("mode") or "").strip().lower() or "live"
    auto_install = body.get("auto_install", None)  # None ⇒ לפי ברירות מחדל של המוד
    ORC = _orc_for_mode(mode, auto_install=auto_install)
    # IntentToSpec לשימוש חוזר בהמשך (נמנע NameError אם planner מצליח)
    i2s = IntentToSpec()

    # ======================================
    # A) Strict grounded answer (no hallucinations)
    # ======================================

    sources = body.get("sources") or _extract_urls(msg)
    # 1) סיווג כוונה ע"י LLM דרך PB
    route = classify_intent(uid, msg, ctx)
    intent, conf = route["intent"], float(route["confidence"])

    # 2) מסלול מידע (Grounded-only)
    if intent == "knowledge":
        if not sources and route.get("needs_sources"):
            return _say(uid, st, "כדי לענות בלי הלוצינציות צריך קישורים/מקורות.")
        prompt = _sanitize_prompt(msg) or "ענה בקצרה על בסיס המקורות"
        out = GW.chat(user_id=uid, task="answer", intent="answer",
                    content={"prompt": prompt, "sources": sources, "context": ctx},
                    require_grounding=True, temperature=float(body.get("temperature", 0.0)))
        if not out.get("ok"):
            raise HTTPException(400, f"grounded-answer-refused: {out.get('error') or 'not_grounded'}")
        payload = out["payload"]
        return _say(uid, st, payload["text"], extra={"citations": payload.get("citations"), "grounded": True})

    # 3) מסלול שיחה (כשאין ביטחון בבנייה/ידע)
    if intent == "talk" or conf < 0.45:
        reply = GW.chat(user_id=uid, task="chat", intent="chat",
                        content={"prompt": msg, "context": ctx},
                        require_grounding=False, temperature=0.3)
        return _say(uid, st, (reply.get("payload") or {}).get("text",""))

    # 4) מסלול בנייה—מילוי חריצים אם חסר
    required = route.get("required_slots") or []
    if required:
        ef = extract_slots(uid, msg, required)
        if ef["missing"]:
            # שאלה אחת–שתיים לפי required
            q = next((rs["question"] for rs in required if rs["name"] in ef["missing"]), "מה העומס הצפוי?")
            # שמירה למצב – אם יש לך state מתקדם
            st.stage = "slot_fill"
            st.missing = ef["missing"]
            st.slots = ef["values"]
            return _say(uid, st, q, extra={"missing": ef["missing"]})


        # ======================================
    # B) Build path: Discovery → Spec → Refine → Analysis → Execute
    # ======================================
    
    diag_llm = GW.diagnose() if "GW" in globals() else {"enabled":False}
    try:
        spec_planner = ORC.analyze(uid, msg, ctx)
    except up.ValidationFailed: 
        # fallback אוטומטי: IntentToSpec → Minimal
        try:
            i2s = IntentToSpec()
            spec_planner = i2s.from_text(uid, msg)
        except Exception:
            spec_planner = {}
        if not isinstance(spec_planner, dict) or not spec_planner.get("components"):
            spec_planner = {
                "title": "Minimal API",
                "summary": "Local minimal scaffold (planner fallback)",
                "components": [{"name":"api","type":"api","tech":["python","fastapi"],"requires":[]}],
                "tests": [{"name":"healthz","kind":"unit","target":"api"}]
            }
        ctx.setdefault("__diagnostics__",{})["llm"] = diag_llm
    except Exception:
        # כל חריגה אחרת בזמן תכנון – ננתח traceback, ננסה פעולות, ונמשיך
        tb = traceback.format_exc()
        rc = parse_traceback(tb)
        apply_actions(rc.get("actions",[]))
        try:
            i2s = IntentToSpec()
            spec_planner = i2s.from_text(uid, msg)
        except Exception:
            spec_planner = {"title":"Minimal API","summary":"Fallback","components":[{"name":"api","type":"api","tech":["python","fastapi"]}]}
        ctx.setdefault("__diagnostics__",{})["planner"] = rc

    # 2) fallback: IntentToSpec (אם המפרט דל מאוד/דומיין לא מזוהה)
    try:
        spec_i2s = i2s.from_text(uid, msg)
    except Exception:
        spec_i2s = {}

    spec_merged = _merge_specs(spec_planner, spec_i2s)

    # 3) שיפור/השלמת SPEC לדומיין (entities/behavior/tests) — JSON-only
    refiner = SpecRefiner()
    spec_refined = refiner.refine_if_needed(uid, spec_merged)

    # 4) חקירת דרישות/אילוצים/חלופות ו־trade-offs (Structured JSON)
    
    analysis = design_arch(uid, spec_refined, ctx)
    if body.get("autogen_assets"):
        auto = _auto_assets(uid, spec_refined, ctx, msg)
        if auto:
            spec_refined["assets"] = (spec_refined.get("assets") or []) + auto

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

    # --- Approvals & Secrets gate (לפני build) ---

    async def _exec_with_runctx(spec: Dict[str, Any]) -> Dict[str, Any]:
        # פותחים ריצה אטומית לכל Build, כותבים Audit לתיקיית הריצה
        with run_context(user=uid) as run:
            audit_dir = Path(run.path) / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            # מעשירים את ה־ctx כך שכל שכבה יודעת את הריצה
            ctx["run_id"] = run.id
            ctx["run_dir"] = run.path
            ctx["audit_path"] = str(audit_dir / "pipeline.jsonl")
            # מריצים Universal Orchestrator על תיקיית הריצה
            return await ORC.execute(spec, workdir=run.path)
    req = required_approvals({"tools": spec_refined.get("tools"), "title": spec_refined.get("title"), "summary": spec_refined.get("summary")})
    missing_tools   = [t for t in (req.get("tools") or [])   if not CONS.has_tool(uid, t)]
    missing_secrets = [s for s in (req.get("secrets") or []) if not CONS.has_secret(uid, s)]

    if missing_tools or missing_secrets:
        # החזר תשובה “ללא כשל” שמבקשת את האישור/הזדהות — הלקוח ישלח /chat/consent
        return {
            "ok": False,
            "needs_approvals": {
                "tools": missing_tools,
                "secrets": missing_secrets,
                "notes": req.get("notes") or [],
            },
            "hint": "קרא ל־POST /chat/consent כדי לאשר התקנות/להזין סודות ולהמשיך."
        }

    # אפשר להפעיל Auto-install לפי דגל זיכרון (לא חובה)
    if CONS.get_flag(uid, "auto_install"):
        os.environ["IMU_AUTO_INSTALL"] = "1"

    # 5) Execute: בנייה אמיתית (כולל lint/compile/pytest אם יש)
    build_threshold = float(os.getenv("IMU_BUILD_INTENT_T", "0.82"))
    decide = {}
    try:
        decide = classify_intent(uid, msg, ctx) or {}
    except Exception:
        decide = {}
    want_build = (
        bool(body.get("force_build")) or
        ((decide.get("intent") == "build") and float(decide.get("confidence", 0.0)) >= build_threshold)
    )
    if want_build:
        # לנתח → להריץ בתוך run_context אטומי, עם persist_dir/audit per-run
        with run_context(user=uid) as run:
            audit_dir = Path(run.path) / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            ctx["run_id"] = run.id
            ctx["run_dir"] = run.path
            ctx["audit_path"] = str(audit_dir / "pipeline.jsonl")
            spec = UNIV.analyze(uid, msg, ctx)
            res  = await UNIV.execute(spec, workdir=run.path)
        extra = {
            "run_id": ctx["run_id"],
            "build": res.get("build"),
            "persist_dir": (res.get("build") or {}).get("persist_dir"),
            "intent": decide
        }
        return JSONResponse({
            "ok": bool(res.get("ok", True)),
            "text": "✅ יצאתי לבנייה. ראה persist_dir ו‑timeline.",
            "extra": extra,
            "run_id": extra["run_id"],
            "build": extra["build"],
            "persist_dir": extra["persist_dir"]
        })
    miss = res.get("still_missing") or []
    auto = bool(body.get("auto_install") or CONS.get_flag(uid, "auto_install"))
    if miss and auto:
        try:
            # נסיון השגה/התקנה אוטומטית (אם tools/auto_install קיים אצלך)
            from tools.auto_install import auto_install_missing
            _ = await auto_install_missing(miss)
            # נסה שוב בנייה אחרי התקנה
            res = await _exec_with_runctx(spec_refined)
        except Exception:
            pass
    elif miss and not auto:
        # שיח לא טכני: לבקש אישור ולהמשיך
        return _say(uid, st,
            "נדרש אישור להתקנה/חיבור של כלים חיצוניים כדי להמשיך.",
            extra={"needs_approvals":{"tools":miss},"hint":"שלח /chat/consent עם auto_install=true"})
    
    #from engine.self_heal.controller import self_heal_once, classify_failure

    #max_attempts = int(os.environ.get("IMU_SELF_HEAL_MAX_ATTEMPTS", "2"))
    #attempt = 0
    #build = res.get("build", {})
    #files = build.get("inputs", {}) or {}

    #while not res.get("ok") and attempt < max_attempts and isinstance(files, dict) and files:
    #    attempt += 1
    #    cls = classify_failure(build)
    #    fixed, note = self_heal_once(spec_refined, files, build, cls)
    #    if not fixed:
    #       break
    #    new_build = await BUILDER.build_python_module(fixed, name="app_generated")
    #    res = {**res, "build": new_build, "ok": bool(new_build.get("ok"))}
    #    if res["ok"]:
    #        break
    #    build = new_build
    #    files = fixed
    
    # --- Autopilot: ריצה → דיאגנוזה → תיקון → ריצה חוזרת (עד N) ---
    AUTOPILOT = True
    max_attempts = int(os.environ.get("IMU_SELF_HEAL_MAX_ATTEMPTS", "2"))
    attempt = 0

    while AUTOPILOT and not res.get("ok") and attempt < max_attempts:
        attempt += 1
        co = (res.get("build",{}) or {}).get("compile_out","") or (res.get("build",{}) or {}).get("test_out","")
        if not co:
            break
        rc = parse_traceback(co)  # ← קורא traceback אמיתי (לא טקסט גנרי)
        files_map = (res.get("build",{}) or {}).get("inputs") or {}
        apply_actions(rc.get("actions", []), files=files_map)  # ← מחיל פעולות תיקון מדויקות
        res = await _exec_with_runctx(spec_refined)                # ← ריצה חוזרת

    ctx.setdefault("__diagnostics__",{}).setdefault("autopilot",{})["attempts"] = attempt

    miss = res.get("still_missing") or []
    auto = bool(body.get("auto_install") or CONS.get_flag(uid, "auto_install"))
    if miss and auto:
        try:
            # נסיון השגה/התקנה אוטומטית (אם tools/auto_install קיים אצלך)
            from tools.auto_install import auto_install_missing
            _ = await auto_install_missing(miss)
            # נסה שוב בנייה אחרי התקנה
            res = await _exec_with_runctx(spec_refined)
        except Exception:
            pass
    elif miss and not auto:
        # שיח לא טכני: לבקש אישור ולהמשיך
        return _say(uid, st,
            "נדרש אישור להתקנה/חיבור של כלים חיצוניים כדי להמשיך.",
            extra={"needs_approvals":{"tools":miss},"hint":"שלח /chat/consent עם auto_install=true"})


    # --- Optional post-build actions for non-technical flow ---
    persist_dir = body.get("persist")
    emit_ci     = bool(body.get("emit_ci"))
    emit_iac    = bool(body.get("emit_iac"))
    autodeploy  = bool(body.get("autodeploy"))
    emit_market = bool(body.get("emit_market_scan"))

    written = []

    # 1) Persist generated files (if requested)
    if persist_dir and isinstance(res.get("build", {}).get("inputs"), dict):
        files_map = res["build"]["inputs"]           # {path: bytes|str}
        for rel, data in files_map.items():
            p = Path(persist_dir) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, str):
                data = data.encode("utf-8")
            p.write_bytes(data)
            written.append(str(p))
    
    # --- Asset Renderer: כתיבת תוצרים כלליים אם ה-Spec כולל 'assets' ---
    assets = (spec_refined.get("assets") or [])
    if persist_dir and isinstance(assets, list):
        for a in assets:
            try:
                rel = str(a.get("path") or "").strip()
                body_txt = str(a.get("body") or "")
                if not rel or not body_txt: 
                    continue
                # בטיחות: אל תאפשר יציאה מהעץ
                if ".." in rel or rel.startswith(("/", "\\")):
                    continue
                p = Path(persist_dir) / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(body_txt.encode("utf-8"))
                written.append(str(p))
            except Exception:
                pass


    # 2) Emit CI workflow
    if emit_ci:
        ci_files = resolve_blueprint("ci.github_actions")(spec_refined)
        for rel, data in ci_files.items():
            p = Path(persist_dir or ".") / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data if isinstance(data, (bytes, bytearray)) else str(data).encode())

    # 3) Emit Terraform IaC
    if emit_iac:
        iac_files = resolve_blueprint("iac.terraform")(spec_refined)
        for rel, data in iac_files.items():
            p = Path(persist_dir or ".") / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data if isinstance(data, (bytes, bytearray)) else str(data).encode())

    # 4) Market scan (grounded) – optional
    if emit_market:
        try:
            ms_files = resolve_blueprint("market.scan")(spec_refined | {"context": ctx})
            for rel, data in ms_files.items():
                p = Path(persist_dir or ".") / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data if isinstance(data, (bytes, bytearray)) else str(data).encode())
        except Exception:
            pass
    
    emit_list = body.get("emit") or []  # למשל ["ui.chat_console"]
    if isinstance(emit_list, list):
        for bp_name in emit_list:
            try:
                bp_files = resolve_blueprint(str(bp_name))(spec_refined)
                for rel, data in bp_files.items():
                    p = Path(persist_dir or ".") / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(data if isinstance(data,(bytes,bytearray)) else str(data).encode())
                    written.append(str(p))
            except Exception:
                pass
        # 5) Autodeploy (safe: return script; not executed unless explicitly allowed)
    deploy_script = []
    if autodeploy:
        deploy_script = [
            "docker compose build && docker compose up -d",
            "cd infra/terraform && terraform init && terraform apply -auto-approve"
        ]
    
    # 6) ניסוח מענה ידידותי למשתמש + החזרת מידע עשיר
    if res.get("ok"):
        text = "בנוי ✔️"
    elif res.get("still_missing"):
        text = "נדרשים כלים חיצוניים (ראה extra.instructions)"
    else:
        text = "הבנייה נכשלה — צירפתי דיאגנוסטיקה והצעתי תיקון"

    from engine.deploy.rollback import suggest_rollback
    rb = suggest_rollback(spec_refined.get("title","app"))
    extra = {
        "mode": mode,
        "spec": spec_refined,
        "analysis": analysis,
        "files_written": written,
        "deploy_script": deploy_script,
        **res,  # build, tools, missing, instructions, blueprint, domain...
        "rollback": rb,
        "__diagnostics__": ctx.get("__diagnostics__", {})
    }
    def _human_text(res, ctx):
        if res.get("ok"):
            return "בנוי ✔️"
        rc = ((ctx.get("__diagnostics__",{}) or {}).get("build_rc") or {})
        cat = rc.get("category")
        if cat == "permission":
            return "תיקנתי נתיבי כתיבה ובניתי שוב (הרצה מקומית בטוחה)."
        if cat == "sandbox_runner_bwrap":
            return "נכשל ראנר bwrap – כיביתי סנדבוקס והרצתי ישיר."
        if cat == "planner_empty_spec":
            return "מודל תכנון לא זמין – נפלתי למסלול מקומי והמשכתי."
        return "הבנייה נכשלה – צירפתי הסבר ודיאגנוזה ב-extra."

    # --- TrustOps KPI ---
    try:
        import time
        import pathlib
        kdir = pathlib.Path(".imu/trustops")
        kdir.mkdir(parents=True, exist_ok=True)
        b = res.get("build",{}) or {}
        kpi = {
            "ts": time.time(), "user": uid, "mode": mode,
            "build_ok": bool(res.get("ok")),
            "compile_rc": b.get("compile_rc"), "test_rc": b.get("test_rc"),
            "blueprint": res.get("blueprint"), "domain": res.get("domain"),
            "files_count": len(b.get("files_built") or []),
            "note": "design is ungrounded by policy; knowledge remains grounded"
        }
        (kdir / f"run_{int(kpi['ts'])}.json").write_text(json.dumps(kpi, ensure_ascii=False, indent=2), "utf-8")
    except Exception:
        pass

    text = _human_text(res, ctx)
    return _say(uid, st, text, extra=extra)

@router.post("/consent")
def consent(body: Dict[str,Any]):
    uid  = (body.get("user_id") or "user").strip()
    tools = body.get("tools") or []
    secrets = body.get("secrets") or {}   # {"OPENAI_API_KEY":"..."} למשל
    auto_install = bool(body.get("auto_install"))
    if tools:
        CONS.grant_tools(uid, [str(t) for t in tools])
    for k,v in secrets.items():
        CONS.put_secret(uid, str(k), str(v))
    CONS.set_flag(uid, "auto_install", auto_install)
    return {"ok": True, "msg": "consents recorded"}

@router.post("/preferences")
def set_preferences(body: Dict[str,Any]):
    uid = (body.get("user_id") or "user").strip()
    tone = (body.get("tone") or {})
    for k,v in tone.items():
        UM.pref_set(uid, f"tone.{k}", v, confidence=0.9)
    return {"ok": True}

@router.post("/memory/reset")
def deep_reset(body: Dict[str, Any]):
    uid   = (body.get("user_id") or "user").strip()
    level = (body.get("level") or "t0").lower()  # "t0" | "deep"
    if level == "t0":
        MB.reset_t0(uid)
        SESS.pop(uid, None)
        return {"ok": True, "cleared": ["t0"]}
    # deep: מוחק פרופיל/פרסונה (t1/t2)
    MB.wipe_user(uid)
    SESS.pop(uid, None)
    return {"ok": True, "cleared": ["t0","t1","t2"]}


@router.post("/say", summary="Say Plain (no JSON)")
async def say_plain(
    message: str = Body(..., media_type="text/plain"),
    user_id: str = Query("u1"),
    mode: str = Query("live"),
    proceed: bool = Query(False),
    autobuild: bool = Query(False),
    persist: str = Query("./out"),
    emit_ci: bool = Query(False),
    emit_iac: bool = Query(False),
    autodeploy: bool = Query(False),
    ignore_persona: bool = Query(True),
    ignore_memory:  bool = Query(True),
):
    return await send({
        "user_id": user_id, "message": message.strip(), "mode": mode,
        "proceed": proceed, "autobuild": autobuild, "persist": persist,
        "emit_ci": emit_ci, "emit_iac": emit_iac, "autodeploy": autodeploy,
        "ignore_persona": ignore_persona, "ignore_memory": ignore_memory
    })



@router.get("/say")
async def say_query(text: str, user_id: str = "u1",
                    mode: str = "live", proceed: bool = False,
                    autobuild: bool = False, persist: str = "./out"):
    """
    קולט טקסט טבעי כפרמטר query – לדפדפן – ומעביר ל-/chat/send.
    """
    body = {
        "user_id": user_id,
        "message": text,
        "mode": mode,
        "proceed": proceed,
        "autobuild": autobuild,
        "persist": persist
    }
    return await send(body)

@router.post("/say-form")
async def say_form(user_id: str = Form("u1"),
                   message: str = Form(...),
                   mode: str = Form("live"),
                   proceed: bool = Form(False),
                   autobuild: bool = Form(False),
                   persist: str = Form("./out")):
    """
    קולט טופס form-encoded (ללא JSON) – ומעביר ל-/chat/send.
    """
    body = {
        "user_id": user_id,
        "message": message,
        "mode": mode,
        "proceed": proceed,
        "autobuild": autobuild,
        "persist": persist
    }
    return await send(body)
