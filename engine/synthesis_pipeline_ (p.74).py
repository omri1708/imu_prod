# imu_repo/engine/synthesis_pipeline.py
from __future__ import annotations
import os, time, json
from typing import Dict, Any
from synth.specs import BuildSpec, Contract
from synth.plan import plan
from synth.generate import generate_sources
from synth.test import run_tests
from synth.verify import EvidenceStore, verify_against_contracts
from synth.package import make_tarball
from synth.canary_multi import run_staged_canary
from synth.rollout import gate
from grounded.provenance import ProvenanceStore
from grounded.fact_gate import require_claims
from grounded.evidence_policy import policy_singleton as EvidencePolicy
from grounded.contract_enforcer import enforce_min_trust
from grounded.http_verifier import fetch_and_record
from grounded.consistency import analyze_consistency
from grounded.contradiction_resolution import resolve_contradictions
from grounded.auto_patch import auto_patch_from_resolution
from grounded.policy_overrides import apply_policy_overrides
from synth.schema_validate import validate
from perf.measure import load_test
from user_model.routing import reorder_lang_pref
from user_model.memory import UserMemory
from user_model.policies import get_effective
from db.sandbox_sqlite import ensure_schema, migrate
from alerts.alerts import evaluate as eval_alerts
from ui.accessibility import analyze_ui_folder
from sandbox.fs_net import NetSandbox
from sandbox.session_rl import SessionLimiter
from kpi.policy_adapter import compute_kpi_with_policy
from kpi.score import resolution_quality_from_proof
from guard.anti_regression import check_and_record as anti_reg_check
from grace.grace_manager import grant as grace_grant, active as grace_active
from user_model.consolidation import Consolidator     
from user_model.emotion import detect as detect_emotion  
from engine.plugin_registry import run_plugins  
from grounded.api_gate import OfficialAPIGate, ApiGateError 
from engine.gates.min_evidence import MinEvidenceGate 
from engine.user_policy_bridge import apply_user_policy
from engine.gates.user_conflict_gate import UserConflictGate  
from user_model.memory_store import get_profile  
from engine.runtime_bridge import apply_runtime_gates  
from runtime.metrics import metrics  

from __future__ import annotations
import json, hashlib, time
from typing import Dict, Any

from grounded.claims import _current
from engine.capability_wrap import guard_text_capability_for_user
from synth.validators import (
    validate_spec, validate_plan, validate_generate,
    run_unit_tests, run_integration_tests, run_verify,
    validate_tests, validate_verify, validate_package
)

def _check_prov_sig_fresh(pv: ProvenanceStore, keys):
    for k in keys:
        rec = pv.get(k)
        if not rec or not rec.get("_sig_ok", False):
            raise RuntimeError(f"provenance_sig_invalid:{k}")
        if not rec.get("_fresh", False):
            raise RuntimeError(f"provenance_stale:{k}")

def run_pipeline(spec: BuildSpec, out_root: str="/mnt/data/imu_builds", user_id: str="anon") -> Dict[str,Any]:
    os.makedirs(out_root, exist_ok=True)

    # 0) טענת מדיניות פר־משתמש/אפליקציה והחלה על המנועים
    effective_policy = get_effective(user_id, spec.name)
    apply_policy_overrides(effective_policy)
    extras = getattr(spec, "extras", {}) if hasattr(spec, "extras") else {}
    
    # 0.1) איפוס מטריקות ריצה (אופציונלי דרך קונפיג)
    try:
        if (extras.get("runtime_metrics") or {}).get("reset_before_run", False):
            metrics.reset()
    except Exception:
        pass
 
    base_targets = {"p95_ms": 1500.0, **(effective_policy.get("targets") or {})}
    try:
        base_targets = apply_user_policy(user_id, base_targets) or base_targets
    except Exception:
        pass  # נשאר עם ברירת מחדל/targets מהמדיניות אם אין פרופיל משתמש

    spec.language_pref = reorder_lang_pref(user_id, spec.language_pref or [])
    dag = plan(spec)
    build_dir = os.path.join(out_root, f"{int(time.time()*1000)}_{spec.name}")
    os.makedirs(build_dir, exist_ok=True)

 
    # 0.5) ── נמשוך פרופיל T2 (אם קיים) ונעדכן העדפות שפה/מדדים מראש ──
    cons = Consolidator()
    snap = cons.snapshot(user_id)
    lang_pref = (snap.get("profile", {}).get("preferences", {}).get("lang_pref", {}) or {}).get("value")
    if lang_pref:
        spec.language_pref = [lang_pref] + [x for x in (spec.language_pref or []) if x != lang_pref]


    # 1) generate
    gen = generate_sources(spec, os.path.join(build_dir, "src"))
    
    # 1.1) plugins
    plugin_out = run_plugins(spec, build_dir, user_id)
    plugin_evidence = plugin_out.get("evidence", {})
    plugin_kpi_score = float(plugin_out.get("kpi", {}).get("score", 0.0))
    
    # 1.2) official API checks (אחרי plugins ולפני tests)
    official_evs = []
    try:
        extras = getattr(spec, "extras", {}) or {}
        official_checks = extras.get("official_api_checks") or []
        if official_checks:
            gate = OfficialAPIGate()
            for chk in official_checks:
                res = gate.verify(
                    name        = chk["name"],
                    url         = chk["url"],
                    json_schema = chk["schema"],
                    claim_path  = chk["claim_path"],
                    expected    = chk["expected"],
                    user_id     = user_id,
                    obj         = spec.name,
                    tags        = ["pipeline"]
                )
                # שומרים רק את רשומת הראיה (ledger record)
                official_evs.append(res["evidence"])
    except ApiGateError as e:
        raise RuntimeError(f"official_api_check_failed:{e}")

    # 2) tests
    checks=[{"path":"/health","expect_status":200,"expect_contains":"OK"},
            {"path":"/ui","expect_status":200,"expect_contains":"IMU App"}]
    for p in spec.endpoints:
        checks.append({"path":p,"expect_status":200,"expect_contains":""})
    t_res = run_tests(gen["language"], gen["entry"], checks)

    # 3) perf
    paths = [c["path"] for c in checks]
    perf = load_test(t_res["port"], paths, concurrency=8, total_requests=50)
    # Gate לפי יעד p95_ms שנקבע בשלב 0
    _target_ms = float(base_targets.get("p95_ms", 1500.0))
    perf_gate = {"ok": perf["p95_ms"] <= _target_ms,
                 "target_ms": _target_ms,
                 "actual_ms": float(perf["p95_ms"])}

    # 4) DB bootstrap + migration
    db_path = os.path.join(build_dir, "sandbox.db")
    ensure_schema(db_path, "CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT, created REAL);")
    mig_out = migrate(db_path, [
        "ALTER TABLE notes RENAME TO notes_v1",
        "CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT, created REAL, author TEXT)",
        "INSERT INTO notes(body,created,author) SELECT body,created,'system' FROM notes_v1",
        "DROP TABLE notes_v1",
        "INSERT INTO notes(body,created,author) VALUES('hello', strftime('%s','now'), 'imu')",
        "SELECT COUNT(*) AS n FROM notes"
    ])

    # 5) UI evidence
    ui_ev = analyze_ui_folder(os.path.join(build_dir,"src","ui"))

    # 6) evidence stores
    evidence_dir = os.path.join(build_dir,"evidence")
    es = EvidenceStore(evidence_dir)
    pv = ProvenanceStore(evidence_dir)
    collected_evidence: Dict[str, Any] = {}
    es.put("service_tests", t_res);             collected_evidence["service_tests"] = t_res
    es.put("perf_summary", perf);               collected_evidence["perf_summary"]  = perf
    es.put("user_targets", base_targets);       collected_evidence["user_targets"] = base_targets
    es.put("perf_target_gate", perf_gate);      collected_evidence["perf_target_gate"] = perf_gate
    es.put("db_migration", {"out": mig_out});   collected_evidence["db_migration"]  = {"out": mig_out}
    es.put("ui_accessibility", ui_ev);          collected_evidence["ui_accessibility"] = ui_ev
       # הזרקת ראיות מה-plugins למחסני הראיות
    es.put("plugin_evidence", plugin_evidence); collected_evidence["plugin_evidence"] = plugin_evidence
    if official_evs:
        es.put("official_api", {"checks": official_evs})
    pv.put("service_tests", t_res, source_url="internal.test://evidence", trust=0.99)
    pv.put("perf_summary",  perf, source_url="internal.test://evidence", trust=0.99)
    pv.put("user_targets", base_targets, source_url="internal.policy://user_targets", trust=0.90)
    pv.put("perf_target_gate", perf_gate, source_url="internal.policy://user_targets", trust=0.95)
    pv.put("db_migration", {"out": mig_out}, source_url="internal.test://evidence", trust=0.99)
    pv.put("ui_accessibility", ui_ev, source_url="internal.test://evidence", trust=0.95)
    if plugin_evidence:
        pv.put("plugin_evidence", plugin_evidence, source_url="internal.plugin://run", trust=0.90)
    if official_evs:
        pv.put("official_api", {"checks": official_evs}, source_url="internal.official_api://verify", trust=0.99)
        collected_evidence["official_api"] = {"checks": official_evs}
    
    # 7) external evidence (אופציונלי)
    ns = NetSandbox(max_bytes=2_000_000)
    sess_rl = SessionLimiter()
    for ev in (spec.external_evidence or []):
        k = ev.get("key"); url = ev.get("url")
        if not k or not url: continue
        if not sess_rl.allow(user_id, 4096):
            raise RuntimeError("session_rate_limited")
        fetch_and_record(k, url, pv)

    cons.add_event(user_id, "preference", {"key":"lang_pref","value": gen["language"]}, confidence=0.8, trust=0.9, stable_hint=True)
        # ── קונסולידציה בסוף הריצה: קידום T0/T1 → T2 ──
    cons_out = cons.consolidate(user_id)
    
    # 8) gates בסיסיים
    req_keys = list(spec.evidence_requirements or [])
    for ev in (spec.external_evidence or []):
        if ev.get("key") and ev["key"] not in req_keys:
            req_keys.append(ev["key"])
    _check_prov_sig_fresh(pv, req_keys)
    EvidencePolicy.check(pv, req_keys)

    # 9) Consistency + Resolution עם trust_cut מהמדיניות
    cons = analyze_consistency(pv, req_keys)
    resolution_score = 100.0
    if not cons["ok"]:
        res = resolve_contradictions(pv, req_keys,
                                     trust_cut=float(effective_policy.get("trust_cut_for_resolution", 0.80)),
                                     method="wmedian")
        if not res.ok:
            raise RuntimeError(f"evidence_inconsistent_unresolved: score={cons['score']:.1f}")
        auto_patch_from_resolution(
            build_dir, pv,
            {"ok": res.ok, "effective": res.effective, "used": res.used, "dropped": res.dropped, "proof": res.proof},
            tighten_trust_by_key=effective_policy.get("min_trust_by_key") or {},
            min_consistency_score=effective_policy.get("min_consistency_score")
        )
        pv.put("resolved_metrics", {"effective": res.effective}, source_url="internal.test://resolution", trust=0.99)
        es.put("resolved_metrics", {"effective": res.effective})
        collected_evidence["resolved_metrics"] = {"effective": res.effective}
        if "resolved_metrics" not in req_keys: req_keys.append("resolved_metrics")
        from kpi.score import resolution_quality_from_proof
        resolution_score = resolution_quality_from_proof(
            contradictions_after_cut=len(res.proof.get("contradictions_after_cut", [])),
            base_score=float(res.proof.get("consistency_score", 100.0))
        )
        cons["score"] = float(res.proof.get("consistency_score", cons["score"]))

    # 10) contracts: סכימות + min_trust per-contract
    all_ok=True; violations=[]
    bundle = {"tests": t_res, "perf": perf, "ui": ui_ev}
    for c in spec.contracts:
        if c.schema:
            ok, errs = validate(bundle, c.schema)
            if not ok: all_ok=False; violations += [f"{c.name}:{e}" for e in errs]
        if c.evidence_min_trust:
            enforce_min_trust(pv, req_keys, c.evidence_min_trust)
    base_ver = verify_against_contracts([{"name": cc.name, "schema": cc.schema} for cc in spec.contracts], t_res)
    all_ok = all_ok and base_ver["ok"]
    ver = {"ok": all_ok, "violations": violations + base_ver.get("violations",[])}

    min_ev_cfg = getattr(spec, "extras", {}).get("min_evidence_gate") if hasattr(spec,"extras") else None
    if min_ev_cfg:
        gate = MinEvidenceGate(min_ev_cfg.get("kinds", []), min_ev_cfg.get("min", 1))
        res  = gate.check(collected_evidence)
        if not res["ok"]:
            raise RuntimeError(f"min_evidence_failed:found={res['found']}/need={res['need']};present={res['present']}")
    # 11) KPI לפי משקולות המדיניות (+ שילוב תוספי plugins)
    base_kpi = compute_kpi_with_policy(
        tests_passed=t_res["passed"],
        p95_ms=perf["p95_ms"],
        ui_score=ui_ev.get("score", 0),
        consistency_score=cons["score"],
        resolution_score=resolution_score,
        weights=(effective_policy.get("kpi_weights") or {})
    )
    # 11.1)    # משקל אופציונלי לפלאגינים מתוך המדיניות (ברירת מחדל 0)
    _weights = effective_policy.get("kpi_weights") or {}
    plugins_w = float(_weights.get("plugins", 0.0))
    if plugins_w > 0.0:
        final_score = base_kpi["score"] * (1.0 - plugins_w) + plugin_kpi_score * plugins_w
        kpi = {**base_kpi, "score": final_score,
               "parts": [{"name":"policy_base","score":base_kpi["score"]},
                         {"name":"plugins","score":plugin_kpi_score,"weight":plugins_w}]}
    else:
        kpi = {**base_kpi, "parts": [{"name":"policy_base","score":base_kpi["score"]},
                                     {"name":"plugins","score":plugin_kpi_score,"weight":0.0}]}

    # 11.2) UserConflictGate (לפני החלטת ה-rollout)
    
    ucg_cfg = (getattr(spec, "extras", {}) or {}).get("user_conflict_gate")
    if ucg_cfg and user_id:
        prof = get_profile(user_id)  # ← מגיע מ-memory_store, בפורמט {pref, beliefs, strength}
        gate_ucg = UserConflictGate(
            keys=ucg_cfg.get("keys", []),
            max_ambiguity=ucg_cfg.get("max_ambiguity", 0.2),
            min_strength=ucg_cfg.get("min_strength", 0.5),
        )
        ucg_res = gate_ucg.check(prof)
        # רישום כראיה
        es.put("user_conflict_check", ucg_res)
        pv.put("user_conflict_check", ucg_res, source_url="internal.user://profile", trust=0.85)
        collected_evidence["user_conflict_check"] = ucg_res
        if not ucg_res["ok"]:
            raise RuntimeError(f"user_conflict_gate_failed:{ucg_res['offenders']}")
    # 11.3) Runtime gates (לפני החלטת ה-rollout הסופית)
    try:
        rb = apply_runtime_gates(extras) or {}
    except Exception as e:
        rb = {"ok": False, "error": f"runtime_bridge_failed:{e}"}
    es.put("runtime_bridge", rb)
    pv.put("runtime_bridge", rb, source_url="internal.runtime://bridge", trust=0.9)
    collected_evidence["runtime_bridge"] = rb
    # אכיפה אופציונלית (אם תרצה לחסום rollout כש־rb["ok"]=False)
    if (extras.get("runtime_gates") or {}) and extras.get("runtime_gate_enforce", False):
        if rb.get("ok") is False:
            raise RuntimeError("runtime_gate_failed")
    
    # 12) Canary מרובה־שלבים — לפי policy אם קיים
    stages = effective_policy.get("canary_stages")
    baseline_kpi = 75.0
    from synth.canary_multi import run_staged_canary
    staged = run_staged_canary(baseline_kpi=baseline_kpi, candidate_kpi=kpi["score"], stages=stages)

    # 13) Anti-Regression — לפי policy
    ar = effective_policy.get("anti_regression") or {}
    anti = anti_reg_check(
        service=spec.name, kpi_score=kpi["score"], p95_ms=perf["p95_ms"],
        max_allowed_regression_pct=float(ar.get("max_regression_pct", 7.5)),
        min_allowed_kpi=float(ar.get("min_kpi", 70.0))
    )
    grace_info = None
    if not anti["ok"]:
        if not grace_active(user_id):
            g = grace_grant(user_id, reason=anti["reason"], ttl_s=1800)
            grace_info = g
            if not g["ok"]:
                staged["approved"] = False
                staged["reason"] = f"anti_regression:{anti['reason']} (no grace)"
        # אם יש גרייס — נדרש שלבי shadow/1% לעבור:
        st_ok = all(s["ok"] for s in staged["stages"] if s["stage"] in ("shadow", "1pct"))
        if not st_ok:
            staged["approved"] = False
            staged["reason"] = f"anti_regression:{anti['reason']} (grace but early stages failed)"
        # אם יעד p95 של המשתמש הופר — לא מאשרים rollout
    
    if not perf_gate["ok"]:
        staged["approved"] = False
        staged["reason"] = f"p95_target_exceeded:{perf_gate['actual_ms']}>{perf_gate['target_ms']}"

    roll = {"approved": bool(staged["approved"]), "staged": staged, "kpi": kpi, "anti_regression": anti, "grace": grace_info}

    alerts = eval_alerts({"perf":perf,"verify":ver}, build_dir)

    mem = UserMemory()
    mem.put_episode(user_id, "preference", {"key":"lang_pref","value": gen["language"]}, confidence=0.8)
    mem.consolidate(user_id)

    artifact = make_tarball(os.path.join(build_dir,"src"), os.path.join(build_dir, "artifact.tar.gz"))
    summary = {
        "generated": gen,
        "tests": t_res, "perf": perf, "db_migration": mig_out, "ui": ui_ev,
        "consistency": cons, "verify": ver, "artifact": artifact,
        "rollout": roll, "alerts": alerts,
        "required_evidence": req_keys, "evidence_dir": evidence_dir,
        "policy": effective_policy,
        "targets": {"base": base_targets, "perf_gate": perf_gate},
        "user_profile": cons_out.get("profile", {}),
        "plugins": {
            "list": list(plugin_out.get("evidence", {}).keys()),
            "kpi_score": plugin_kpi_score
        }
    }
    with open(os.path.join(build_dir,"summary.json"),"w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

# TODO- הערה: ההזרקה הדגמתית של רגש משתמשת בטקסט קשיח רק כדי להוכיח זרימה — במערכת שלך חבר זאת ל־NLU של השיחה (או ל־UI). אין תלות חיצונית.