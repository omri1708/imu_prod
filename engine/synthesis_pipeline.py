# imu_repo/engine/synthesis_pipeline.py
from __future__ import annotations
import json, hashlib, time
from typing import Dict, Any, Optional
from contextlib import suppress

from engine.ab_selector import select_best
from engine.learn import learn_from_pipeline_result
from engine.learn_store import load_baseline, _task_key, load_history
from engine.config import load_config
from engine.explore_policy_ctx import decide_explore_ctx
from engine.explore_state import mark_explore, mark_regression, clear_regression
from engine.provenance_gate import enforce_evidence_gate, GateFailure
from engine.capability_wrap import text_capability_for_user

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
    policy = PolicyEngine(cfg.get("policy"))
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
            policy_engine=policy, min_trust=cfg_min_trust
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
    # --- Negative Guard לפני אריזה/חתימה (Safe-Progress) ---
    try:
        # נרצה לכלול גם את הראיה של ה-package: אוספים שוב את הראיות
        evs_for_guard = _collect_evidence_preserving_buffer()
        candidate_page = {
            "title": spec.get("name", "artifact"),
            "body": txt_pkg["artifact_text"],
            "lang": txt_pkg.get("lang") or txt_pkg.get("language") or "text/plain",
        }
        guard_res = run_negative_suite(candidate_page, evs_for_guard, policy=gate_before.get("policy"))
        record_event("rollout_guard_pass", {
            "checked": guard_res.get("checked"),
            "sources": guard_res.get("sources"),
            "agg_trust": guard_res.get("agg_trust"),
        }, severity="info")
    except RolloutBlocked as rb:
        record_event("rollout_guard_block", {"reason": str(rb)}, severity="error")
        with suppress(Exception):
            cooldown_s = float(cfg.get("explore", {}).get("cooldown_s", 900.0))
            mark_regression(key, cooldown_s=cooldown_s)
        raise
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
            "manifest_sha": signed_pkg["manifest_sha"],
            "artifact_sha": signed_pkg["artifact_sha"],
            "agg_trust": eff_min_trust
        }, severity="info")

    # אם מוגדר IMU_FINGERPRINT_URL ישלח HTTP; אחרת outbox לקובץ
    with suppress(Exception):
        report_fingerprint({"_type": "manifest_link",
                            "manifest_sha": signed_pkg["manifest_sha"]})
    
    # --- Evidence Gate (2): אחרי החבילה (כולל הראיה של החבילה) ---
    try:
        evs_after = _collect_evidence_preserving_buffer()
        gate_after = enforce_evidence_gate(
            evs_after,
            domain=domain, risk_hint=risk_hint,
            policy_engine=policy, min_trust=eff_min_trust
        )
    except GateFailure:
        with suppress(Exception):
            cooldown_s = float(cfg.get("explore", {}).get("cooldown_s", 900.0))
            mark_regression(key, cooldown_s=cooldown_s)
        raise

    # --- Guarded emit (רק אחרי שני ה-Gates וה-Guard) --
    async def _emit_text(_: Dict[str,Any]) -> str:
        return txt_pkg["artifact_text"]
    guarded = await text_capability_for_user(_emit_text, user_id=user_id)
    out = await guarded({"ok": True})

    if learn:
         learn_from_pipeline_result(spec, winner, user_id=user_id)

    # החזרה מורחבת (אם ה-API שלך מאפשר):
    # אפשר להחזיר גם gate info ו-package חתום.
    return {"ok": True, "text": out, "pkg": signed_pkg}