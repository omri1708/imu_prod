# imu_repo/engine/rollout_guard.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

from ui.introspect import extract_ui_claims
from ui.schema_extract import extract_table_specs

from grounded.consistency import (
    check_ui_consistency, ConsistencyError,
    MissingEvidence, ExpiredEvidence, LowTrust, NotEnoughSources,
)
from grounded.schema_consistency import (
    check_table_schema, SchemaError,
)

from engine.runtime_guard import check_runtime_table, RuntimeBlocked
from engine.kpi_regression import gate_from_files, KPIRegressionBlocked
from engine.audit_log import record_event
from engine.auto_remediation import diagnose, propose_remedies, apply_remedies

class RolloutBlocked(Exception):
    ...


def run_negative_suite(
    page_obj: Any,
    evidences: List[Dict[str, Any]],
    *,
    policy: Dict[str, Any],
    runtime_fetcher: Optional[Any] = None,   # ← חדש: הזרקת fetcher לבדיקות
) -> Dict[str, Any]:
    """
    Negative Guard מלא לפני rollout:
      1) עקיבות Claims↔Evidence כללית (min_trust/min_sources)
      2) עקיבות סכימת טבלאות מול evidences (min_schema_sources/min_schema_trust)
      3) Runtime checks + Drift per-table (prev_content_hash מ־policy)
      4) KPI Regression Gate (אם ניתנו קבצים במדיניות)
    """
    # ---- מדיניות ברירת־מחדל ----
    min_trust = float(policy.get("min_trust", 0.75))
    min_sources = int(policy.get("min_sources", 2))
    min_schema_sources = int(policy.get("min_schema_sources", max(2, min_sources)))
    min_schema_trust = float(policy.get("min_schema_trust", min_trust))

    # ---- 1) עקיבות כללית Claims ↔ Evidence ----
    ui_claims = extract_ui_claims(page_obj)
    try:
        res_general = check_ui_consistency(
            ui_claims, evidences,
            min_trust=min_trust,
            min_sources=min_sources,
        )
    except MissingEvidence as e:
        raise RolloutBlocked(f"missing_evidence: {e}")
    except ExpiredEvidence as e:
        raise RolloutBlocked(f"expired_evidence: {e}")
    except NotEnoughSources as e:
        raise RolloutBlocked(f"not_enough_sources: {e}")
    except LowTrust as e:
        raise RolloutBlocked(f"low_trust: {e}")
    except ConsistencyError as e:
        raise RolloutBlocked(f"consistency_error: {e}")

    # ---- 2) סכימה לטבלאות ----
    table_specs = extract_table_specs(page_obj) or []
    schema_tables: List[Dict[str, Any]] = []
    for spec in table_specs:
        table_id = spec.get("binding_url") or spec.get("path") or spec.get("name") or "<unknown>"
        try:
            sch = check_table_schema(
                spec, evidences,
                min_schema_sources=min_schema_sources,
                min_schema_trust=min_schema_trust,
            )
            schema_tables.append({"table": table_id, **sch})
        except SchemaError as e:
            record_event("schema_guard_block", {"table": table_id, "reason": str(e)}, severity="error")
            raise RolloutBlocked(f"schema_error: {e}")

    # ---- 3) Runtime + Drift per-table ----
    runtime_checked = 0
    runtime_sampled = 0
    runtime_tables: List[Dict[str, Any]] = []
    # Auto-Remediation policy
    auto = policy.get("auto_remediation", {}) or {}
    auto_enabled   = bool(auto.get("enabled", False))
    allowed_levels = set(auto.get("apply_levels", ["conservative"]))  # או גם "risky"
    max_rounds     = int(auto.get("max_rounds", 1))

    if bool(policy.get("runtime_check_enabled", True)):
        prev_map = policy.get("runtime_prev_hash_map") or policy.get("prev_hash_map") or {}
        fetcher = runtime_fetcher or policy.get("runtime_fetcher")
        block_on_drift  = bool(policy.get("block_on_drift", False))

        for spec in table_specs:
            table_id = spec.get("binding_url") or spec.get("path") or spec.get("name") or "<unknown>"

            # החדרת baseline ספציפי לטבלה (אם קיים)
            rounds = 0
            while True:
                eff_policy = dict(policy)
                if isinstance(prev_map, dict) and table_id in prev_map:
                    eff_policy["prev_content_hash"] = prev_map[table_id]
                eff_policy["block_on_drift"] = block_on_drift
                try:
                    rrt = check_runtime_table(spec, policy=eff_policy, fetcher=fetcher)
                    runtime_checked += int(rrt.get("checked") or 0)
                    runtime_sampled += int(rrt.get("sampled") or 0)
                    changed = None
                    if "prev_content_hash" in eff_policy and "hash" in rrt:
                        changed = (eff_policy["prev_content_hash"] != rrt["hash"])

                    # אם המדיניות מאפשרת “accept new hash”, עדכן מפה פר-טבלה
                    if changed and bool(policy.get("allow_update_prev_hash_on_schema_ok", False)):
                        mp = policy.setdefault("runtime_prev_hash_map", {})
                        if rrt.get("hash"):
                            mp[table_id] = rrt["hash"]
                            prev_map = mp  # שישתקף בלולאה
                            record_event("runtime_baseline_updated",
                                         {"table": table_id, "new_hash": rrt["hash"]},
                                         severity="warn")

                    runtime_tables.append({
                        "table": table_id,
                        "hash": rrt.get("hash"),
                        "sampled": rrt.get("sampled", 0),
                        "checked": rrt.get("checked", 0),
                        "changed": changed,
                        "rounds": rounds,
                    })
                    record_event("runtime_guard_pass",
                                 {"table": table_id, "hash": rrt.get("hash"),
                                  "sampled": rrt.get("sampled"), "checked": rrt.get("checked"),
                                  "changed": changed, "rounds": rounds},
                                 severity="info")
                    break
                except RuntimeBlocked as rb:
                    record_event("runtime_guard_block",
                                 {"table": table_id, "reason": str(rb), "round": rounds},
                                 severity="error")
                    if not auto_enabled or rounds >= max_rounds:
                        raise RolloutBlocked(f"runtime_error: {rb}") from rb
                    # ריפוי אוטומטי פר-טבלה: מאבחנים ומיישמים רק על הטבלה הנוכחית
                    diags    = diagnose(rb)
                    remedies = [r for r in propose_remedies(diags, policy=policy, table_specs=[spec])
                                if r.safety in allowed_levels]
                    if not remedies:
                        raise RolloutBlocked(f"runtime_error: {rb}")
                    apply_remedies(remedies, policy=policy, table_specs=[spec])
                    record_event("auto_remediation_applied",
                                 {"table": table_id, "remedies": [r.description for r in remedies],
                                  "round": rounds},
                                 severity="warn")
                    rounds += 1
                    continue

    # ---- 4) KPI Regression gate (אופציונלי) ----
    kpi_summary = None
    base_path = policy.get("kpi_baseline_path")
    cand_path = policy.get("kpi_candidate_path")

    if base_path and cand_path:
        try:
            kpi_summary = gate_from_files(base_path, cand_path, policy=policy)
            record_event("kpi_regression_ok", kpi_summary, severity="info")
        except KPIRegressionBlocked as kb:
            record_event(
                "kpi_regression_block",
                {"baseline": base_path, "candidate": cand_path, "reason": str(kb)},
                severity="error",
            )
            raise RolloutBlocked(f"kpi_regression: {kb}") from kb

    return {
        "ok": True,
        "general": res_general,
        "schema": {
            "tables": schema_tables,
            "min_sources": min_schema_sources,
            "min_trust": min_schema_trust,
        },
        "runtime": {
            "sampled": runtime_sampled,
            "checked": runtime_checked,
            "tables": runtime_tables,
        },
        "kpi": kpi_summary,
    }
