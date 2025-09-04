# imu_repo/engine/synthesis_pipeline.py
from __future__ import annotations
import time
import json, hashlib, time
from pathlib import Path    
from typing import Dict, Any, Optional, List
from contextlib import suppress

from engine.ab_selector import select_best
from engine.learn import learn_from_pipeline_result
from engine.learn_store import load_baseline, _task_key, load_history
from engine.config import load_config
from engine.explore_policy_ctx import decide_explore_ctx
from engine.explore_state import mark_explore, mark_regression, clear_regression
from engine.provenance_gate import enforce_evidence_gate, GateFailure
from engine.pipeline_respond_hook import pipeline_respond

from synth.validators import validate_spec, validate_plan, validate_package
from synth.generate_ab import generate_variants
from synth.generate_ab_prior import generate_variants_with_prior
from synth.generate_ab_explore import generate_variants_with_prior_and_explore
from user_model.intent import infer_intent
from grounded.claims import current
from policy.policy_engine import PolicyEngine
from ui.package import build_ui_artifact
from engine.audit_log import record_event
from security.fingerprint_report import report_fingerprint
from engine.rollout_guard import run_negative_suite, RolloutBlocked
from engine.auto_remediation import diagnose, propose_remedies, apply_remedies
from ui.schema_extract import extract_table_specs
from ui.schema_compose import apply_table_specs
from ui.render import render_html
from engine.metrics.jsonl import append_jsonl

def _hash(obj: Any) -> str:
    import hashlib, json as _json
    return hashlib.sha256(_json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

def plan(spec: Dict[str,Any]) -> Dict[str,Any]:
    ok = bool(spec and "name" in spec and "goal" in spec)
    current().add_evidence("spec", {
        "source_url": "local://spec",
        "trust": 0.95 if ok else 0.4,
        "ttl_s": 600,
        "payload": {"ok": ok, "sha256": _hash(spec) if ok else None}
    })
    if not ok:
        raise ValueError("invalid_spec")
    steps = [{"name":"generate_ab[prior/explore_ctx]"},
             {"name":"ab_select_ctx_pareto"},
             {"name":"package"}]
    plan_obj = {"steps": steps, "meta": {"created_at": time.time()}}
    current().add_evidence("plan", {
        "source_url": "local://plan",
        "trust": 0.95,
        "ttl_s": 600,
        "payload": {"sha256": _hash(plan_obj)}
    })
    return plan_obj

def _package_text(spec: Dict[str,Any], winner: Dict[str,Any]) -> Dict[str,Any]:
    label = winner["winner"]["label"]
    artifact = {
        "artifact_name": f"{spec['name']}.txt",
        "lang": winner["winner"]["language"],
        "artifact_text": f"[ARTIFACT:{spec['name']}]\nGOAL={spec['goal']}\nVARIANT={label}\n"
    }
    ok = bool(artifact["artifact_text"])
    current().add_evidence("package",{
        "source_url":"local://package",
        "trust": 0.95 if ok else 0.2,
        "ttl_s": 600,
        "payload":{"ok":ok,"sha256":_hash(artifact)}
    })
    if not ok:
        raise AssertionError("package_invalid")
    return artifact

async def run_pipeline(
    spec: Dict[str,Any], *,
    user_id: str,
    learn: bool = False,
    domain: Optional[str] = None,
    risk_hint: Optional[str] = None,
) -> Dict[str,Any]:
    
    p = plan(spec)
    cfg = load_config()
    policy_engine = PolicyEngine(cfg.get("policy"))
    cfg_min_trust = float(cfg.get("min_trust", 0.75))
    domain = domain or cfg.get("domain")
    risk_hint = risk_hint or cfg.get("risk")
    key = _task_key(str(spec["name"]), str(spec["goal"]))
    baseline = load_baseline(key)
    intents = infer_intent(spec)
    hist = load_history(key, limit=500)

    # --- בחירה אם לבצע Explore אדפטיבית ---
    want_explore = False
    if baseline is not None:
        want_explore = decide_explore_ctx(key=key, intents=intents, history_len=len(hist), cfg=cfg)

    # --- יצירת וריאנטים ---
    if baseline is not None:
        if want_explore:
            variants = generate_variants_with_prior_and_explore(spec, baseline)
            mark_explore(key)
            current().add_evidence("generate_ab_prior_explore_ctx",{
                "source_url":"local://generate_ab_prior_explore_ctx",
                "trust":0.95,"ttl_s":900,
                "payload":{"labels":[v["label"] for v in variants], "intents": intents}
            })
        else:
            variants = generate_variants_with_prior(spec, baseline)
            current().add_evidence("generate_ab_prior",{
                "source_url":"local://generate_ab_prior",
                "trust":0.95,"ttl_s":900,
                "payload":{"labels":[v["label"] for v in variants], "intents": intents}
            })
    else:
        variants = generate_variants(spec)
        current().add_evidence("generate_ab_cold",{
            "source_url":"local://generate_ab_cold",
            "trust":0.95,"ttl_s":900,
            "payload":{"count": len(variants), "labels":[v["label"] for v in variants], "intents": intents}
        })

    # --- בחירת מנצח (Φ+Pareto+Intent+User) ---
    winner = select_best(variants, spec=spec, user_id=user_id, intents=intents)

    # --- סימון רגרסיה לצורך Cool-down ---
    # אם יש baseline וכאשר הנבחר גרוע יותר (phi גבוה יותר) → רגרסיה = הפעלת cooldown
    try:
        phi_new = float(winner["info"]["phi"])
        if baseline is not None:
            base_phi = float(baseline.get("phi", float("inf")))
            if phi_new > base_phi + 1e-9:
                cooldown_s = float(cfg.get("explore", {}).get("cooldown_s", 900.0))
                mark_regression(key, cooldown_s=cooldown_s)
            else:
                # שיפור או שוויון → מפנה רגרסיות קודמות
                clear_regression(key)
    except Exception:
        # לא חוסם את הריצה; במקרה קצה שבו אין מידע — לא נסמן דבר
        pass
    # --- Evidence Gate (1): לפני אריזה/החזרה ---
    def _collect_evidence_preserving_buffer():
        c = current()
        if hasattr(c, "snapshot"):
            return c.snapshot()
        if hasattr(c, "drain"):
            evs = c.drain()
            # נחזיר ל-buffer כדי שלא נאבד ראיות לשלב הבא
            for ev in evs:
                k = ev.get("key", "unknown"); d = dict(ev); d.pop("key", None)
                c.add_evidence(k, d or {})
            return evs
        return []

    try:
        evs_before = _collect_evidence_preserving_buffer()
        gate_before = enforce_evidence_gate(
            evs_before,
            domain=domain, risk_hint=risk_hint,
            policy_engine=policy_engine, min_trust=cfg_min_trust
        )
        # למדיניות יכולה להיות min_trust דינמי:
        eff_min_trust = float(gate_before.get("policy", {}).get("min_trust", cfg_min_trust))
    except GateFailure:
        with suppress(Exception):
            cooldown_s = float(cfg.get("explore", {}).get("cooldown_s", 900.0))
            mark_regression(key, cooldown_s=cooldown_s)
        raise

    # --- יצירת ארטיפקט טקסטי (מוסיף evidence "package") ---
    txt_pkg = _package_text(spec, winner) 
    # --- Negative Guard עם Auto-Remediation לפני חתימה/CAS ---
    # אוספים ראיות מחדש כדי לכלול את ראיית ה-package:
    evs_for_guard = _collect_evidence_preserving_buffer()
    candidate_page = {
        "title": spec.get("name", "artifact"),
        "body": txt_pkg["artifact_text"],
        "lang": txt_pkg.get("lang") or txt_pkg.get("language") or "text/plain",
    }
    policy_cfg = gate_before.get("policy") if isinstance(gate_before, dict) else (cfg.get("policy") or {})
    runtime_fetcher = cfg.get("runtime_fetcher")  # אופציונלי למוקים/בדיקות

    def _finalize_with_auto_remediation(page_obj: Dict[str,Any],
                                        evidences: List[Dict[str,Any]],
                                        policy: Dict[str,Any],
                                        runtime_fetcher=None) -> Dict[str,Any]:
        # מספר ניסיונות לפי מדיניות (דיפולט 3 אם auto_remediation.enabled=True)
        auto = policy.get("auto_remediation", {}) or {}
        attempts = int(auto.get("max_rounds", 3)) if bool(auto.get("enabled", True)) else 1
        last_err: Exception | None = None
        # אם יש לך DSL של טבלאות—נחזיק אותו כדי לאפשר רמדיז משנים (filters/required/sort)
        table_specs = extract_table_specs(page_obj) or []
        for i in range(1, attempts+1):
            try:
                res = run_negative_suite(page_obj, evidences, policy=policy, runtime_fetcher=runtime_fetcher)
                record_event("finalize_guard_ok", {"attempt": i}, severity="info")
                return res
            except RolloutBlocked as rb:
                last_err = rb
                record_event("finalize_guard_blocked", {"attempt": i, "reason": str(rb)}, severity="warn")
                if i >= attempts:
                    break
                # דיאגנוזה ורמדיז — על בסיס החריגה המקורית (אם מוצמדת) או ההודעה
                root = rb.__cause__ or rb
                diags = diagnose(root)
                rems  = propose_remedies(diags, policy=policy, table_specs=table_specs)
                if not rems:
                    break
                apply_remedies(rems, policy=policy, table_specs=table_specs)
                record_event("auto_remediation_applied", {
                    "attempt": i,
                    "remedies": [r.description for r in rems]
                }, severity="warn")
                
                continue
        raise last_err if last_err else RuntimeError("guard_finalize_failed")

    guard_res = _finalize_with_auto_remediation(
        candidate_page, evs_for_guard, policy=policy_cfg, runtime_fetcher=runtime_fetcher
    )
    # שיקוף שינויים ל-UI (אם יש טבלאות בדף)
    composer_mode = (cfg.get("composer") or {}).get("mode", "merge")  # או "overwrite"
    table_specs = extract_table_specs(candidate_page) or []
    if table_specs:
        apply_table_specs(candidate_page, table_specs, mode=composer_mode)

    try:
        current().add_evidence("rollout_guard", {
            "source_url": "local://rollout_guard",
            "trust": 0.95, "ttl_s": 600,
            "payload": {
                "schema": guard_res.get("schema"),
                "runtime": guard_res.get("runtime"),
                "kpi": guard_res.get("kpi")
            }
        })
    except Exception:
        pass
    
    def assert_page_renderable(page_obj) -> None:
        """יזרוק שגיאה אם page_obj לא בפורמט שה-render מקבל."""
        _ = render_html(page_obj, nonce="CHECK") 
    assert_page_renderable(candidate_page)

    # --- יצירת חבילה חתומה/CAS (אחרי שה-Guard עבר) --
    signed_pkg = build_ui_artifact(
        page=candidate_page,        
        nonce=cfg.get("nonce", "IMU_NONCE"),
        key_id=cfg.get("signing_key", "default"),
        cas_root=cfg.get("cas_root"),
        min_trust=eff_min_trust
    )

    # --- Audit + Fingerprint (מיד אחרי יצירת החבילה) ---
    with suppress(Exception):
        record_event("artifact_built", {
            "domain": domain,
            "risk": risk_hint,
            "manifest_sha": (signed_pkg.get("provenance") or {}).get("manifest_sha"),
            "artifact_sha": (signed_pkg.get("provenance") or {}).get("artifact_sha"),
            "agg_trust": eff_min_trust
        }, severity="info")

    # אם מוגדר IMU_FINGERPRINT_URL ישלח HTTP; אחרת outbox לקובץ
    with suppress(Exception):
        report_fingerprint({"_type": "manifest_link",
                            "manifest_sha": (signed_pkg.get("provenance") or {}).get("manifest_sha")})
    
    # --- Evidence Gate (2): אחרי החבילה (כולל הראיה של החבילה) ---
    try:
        evs_after = _collect_evidence_preserving_buffer()
        gate_after = enforce_evidence_gate(
            evs_after,
            domain=domain, risk_hint=risk_hint,
            policy_engine=policy_engine, min_trust=eff_min_trust
        )
    except GateFailure:
        with suppress(Exception):
            cooldown_s = float(cfg.get("explore", {}).get("cooldown_s", 900.0))
            mark_regression(key, cooldown_s=cooldown_s)
        raise

    # --- Respond Hook (אחרי Gate#2) ---
    # בונים הקשר (ctx) עם כל המטא-דאטה הרלוונטי לתגובה
    ctx: Dict[str, Any] = {
        "__policy__": policy_cfg,                      # המדיניות האפקטיבית (כולל מפות drift, KPI וכו')
        "user_id": user_id,
        "domain": domain,
        "risk_hint": risk_hint,
        "task_key": key,
        "spec": spec,
        "guard": guard_res,                            # תוצאת ה-Negative Guard (general/schema/runtime/kpi)
        "gate": {
            "before": gate_before,
            "after":  gate_after
        },
        "package": {
            "sha256": signed_pkg.get("sha256"),
            "provenance": signed_pkg.get("provenance", {}),
            "manifest": signed_pkg.get("manifest", {})
        }
    }
    # הפקת תשובה — agent_emit_answer יאכוף מדיניות/ראיות, ואם חסר – יחסום
    resp = pipeline_respond(ctx=ctx, answer_text=txt_pkg["artifact_text"])
    out = resp  # אם תרצה, תוכל לחלץ כאן שדות ספציפיים מה-Resp
    if learn:
         learn_from_pipeline_result(spec, winner, user_id=user_id)

    # החזרה מורחבת (אם ה-API שלך מאפשר):
    # אפשר להחזיר גם gate info ו-package חתום.

    append_jsonl("/mnt/data/imu_repo/runs/_guard_metrics.jsonl", {
    "ts": time.time(),
    "runtime": guard_res.get("runtime"),
    "kpi": guard_res.get("kpi"),
    "schema": guard_res.get("schema"),
})

    return {
        "ok": True,
        "text": out,                    # מה שנפלט ל-user capability
        "pkg": {                        # סיכום קצר מתוך build_ui_artifact
            "sha256": signed_pkg.get("sha256"),
            "manifest": signed_pkg.get("manifest"),            # המניפסט החתום
            "provenance": signed_pkg.get("provenance", {}),    # כולל artifact_sha / manifest_sha / agg_trust
        },
        "guard": guard_res,             # בדיוק מה שחוזר מ-run_negative_suite (general/schema/runtime/kpi)
        "gate": {
            "before": gate_before,      # תוצאת Evidence Gate #1
            "after":  gate_after        # תוצאת Evidence Gate #2
        }
    }



# === Compatibility shim for tests that call finalize_with_auto_remediation(specs, ...) ===
from typing import Optional
from engine.runtime_guard import check_runtime_table, RuntimeBlocked
from engine.kpi_regression import gate_from_files, KPIRegressionBlocked
from engine.audit_log import record_event
from engine.auto_remediation import diagnose, propose_remedies, apply_remedies

DEFAULT_MAX_ATTEMPTS = 3  

def finalize_with_auto_remediation(
    table_specs: List[Dict[str, Any]],
    *,
    policy: Dict[str, Any],
    runtime_fetcher=None
) -> Dict[str, Any]:
    """גרסת Slim: מריץ Runtime+KPI+Auto-Remediation ישירות על table_specs (לצרכי בדיקות)."""
    attempts = int(policy.get("auto_max_attempts", DEFAULT_MAX_ATTEMPTS)) if bool(policy.get("auto_remediate_enabled", True)) else 1
    last_error: Optional[Exception] = None

    # 1) Runtime עם Auto-Remediation
    if bool(policy.get("runtime_check_enabled", True)):
        for spec in (table_specs or []):
            rounds = 0
            while True:
                try:
                    _ = check_runtime_table(spec, policy=policy, fetcher=runtime_fetcher)
                    record_event("runtime_guard_pass", {"spec": spec.get("binding_url") or spec.get("path")}, severity="info")
                    break
                except RuntimeBlocked as rb:
                    last_error = rb
                    record_event("runtime_guard_block", {"reason": str(rb)}, severity="warn")
                    if rounds >= attempts - 1:
                        return {"ok": False, "attempt": rounds + 1}
                    # אבחון ותיקון
                    diags = diagnose(rb)
                    rems = propose_remedies(diags, policy=policy, table_specs=table_specs)
                    if not rems:
                        return {"ok": False, "attempt": rounds + 1}
                    apply_remedies(rems, policy=policy, table_specs=table_specs)
                    record_event("auto_remediation_applied",
                                 {"remedies": [r.description for r in rems], "round": rounds+1},
                                 severity="warn")
                    rounds += 1
                    continue

    # 2) KPI עם Auto-Remediation קלה (העלאת ספים לפי policy.auto_raise_limits)
    base_path = policy.get("kpi_baseline_path")
    cand_path = policy.get("kpi_candidate_path")
    if base_path and cand_path:
        rounds = 0
        while True:
            try:
                _ = gate_from_files(base_path, cand_path, policy=policy)
                record_event("kpi_regression_ok", {"baseline": base_path, "candidate": cand_path}, severity="info")
                break
            except KPIRegressionBlocked as kb:
                last_error = kb
                record_event("kpi_regression_block", {"reason": str(kb)}, severity="warn")
                if rounds >= attempts - 1:
                    return {"ok": False, "attempt": rounds + 1}
                diags = diagnose(kb)
                rems = propose_remedies(diags, policy=policy, table_specs=table_specs)
                if not rems:
                    return {"ok": False, "attempt": rounds + 1}
                apply_remedies(rems, policy=policy, table_specs=table_specs)
                record_event("auto_remediation_applied",
                             {"scope": "kpi", "remedies": [r.description for r in rems], "round": rounds+1},
                             severity="warn")
                rounds += 1
                continue

    return {"ok": True, "attempt": attempts, "policy": policy, "table_specs": table_specs}







#TODO- אם לא קורה:
# imu_repo/ui/helpers.py
# from __future__ import annotations
# from ui.dsl import Page, Component
# from ui.render import render_html  # לשימוש בבדיקת רנדר

# def page_from_text(title: str, text: str) -> Page:
#    return Page(
#        title=title,
#        components=[Component(kind="markdown", id="artifact_text", props={"md": text})]
#    )

#def assert_page_renderable(page_obj: Page) -> None:
#    # יזרוק חריגה אם page_obj לא תקין ל-render
#    _ = render_html(page_obj, nonce="CHECK")