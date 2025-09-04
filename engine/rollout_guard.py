# imu_repo/engine/rollout_guard.py
from __future__ import annotations
from typing import Dict, Any, List
from ui.introspect import extract_ui_claims
from ui.schema_extract import extract_table_specs
from grounded.consistency import (
    check_ui_consistency, ConsistencyError,
    MissingEvidence, ExpiredEvidence, LowTrust, NotEnoughSources
)
from grounded.schema_consistency import (
    check_table_schema, SchemaError
)
from engine.runtime_guard import check_runtime_table, RuntimeBlocked
from engine.audit_log import record_event

class RolloutBlocked(Exception): ...

def run_negative_suite(
    page_obj: Any,
    evidences: List[Dict[str,Any]],
    *,
    policy: Dict[str,Any]
) -> Dict[str,Any]:
    """
    שלב 1: עקיבות Claims↔Evidence כללית (מקורות/טריות/agg_trust).
    שלב 2: עקיבות סכימה לטבלאות (עמודות/טיפוסים/יחידות).
    """
    # ברירות מחדל מדיניות אם לא נמסרו
    min_trust = float(policy.get("min_trust", 0.75))
    min_sources = int(policy.get("min_sources", 2))
    # לסכימה:
    min_schema_sources = int(policy.get("min_schema_sources", max(2, min_sources)))
    min_schema_trust   = float(policy.get("min_schema_trust", min_trust))

    # 1) עקיבות כללית
    ui_claims = extract_ui_claims(page_obj)
    try:
        res_general = check_ui_consistency(
            ui_claims,
            evidences,
            min_trust=min_trust,
            min_sources=min_sources
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

    # 2) סכימה לטבלאות
    table_specs = extract_table_specs(page_obj)
    for spec in table_specs:
        try:
            res_schema = check_table_schema(
                spec, evidences,
                min_schema_sources=min_schema_sources,
                min_schema_trust=min_schema_trust
            )
        except SchemaError as e:
            raise RolloutBlocked(f"schema_error: {e}")
    # 3) בדיקות Runtime (אופציונלי לפי policy) + Drift/Hash
    runtime_checked = 0
    runtime_sampled = 0
    runtime_tables: List[Dict[str,Any]] = []
    if bool(policy.get("runtime_check_enabled", True)):
        # מפה אופציונלית של prev-hash פר טבלה (URL→hash)
        prev_map = (
            policy.get("runtime_prev_hash_map")
            or policy.get("runtime_prev_hash")
            or policy.get("prev_hash_map")
            or {}
        )
        fetcher = policy.get("runtime_fetcher")
        block_on_drift = bool(policy.get("block_on_drift", False))

        for spec in table_specs:
            table_id = spec.get("binding_url") or spec.get("path") or spec.get("name") or "<unknown>"
            # Policy אפקטיבי פר-טבלה (כדי להזרים prev_content_hash מתאים)
            eff_policy = dict(policy)
            if isinstance(prev_map, dict) and table_id in prev_map:
                eff_policy["prev_content_hash"] = prev_map[table_id]
            eff_policy["block_on_drift"] = block_on_drift

            try:
                rrt = check_runtime_table(spec, policy=eff_policy, fetcher=fetcher)
                runtime_checked += int(rrt.get("checked", 0))
                runtime_sampled += int(rrt.get("sampled", 0))
                # נחשב changed אם יש לנו prev_content_hash אפקטיבי
                changed = None
                if "prev_content_hash" in eff_policy and "hash" in rrt:
                    changed = (eff_policy["prev_content_hash"] != rrt["hash"])
                runtime_tables.append({
                    "table": table_id,
                    "hash": rrt.get("hash"),
                    "sampled": rrt.get("sampled", 0),
                    "checked": rrt.get("checked", 0),
                    "changed": changed,
                })
                record_event(
                    "runtime_guard_pass",
                    {
                        "table": table_id,
                        "hash": rrt.get("hash"),
                        "sampled": rrt.get("sampled"),
                        "checked": rrt.get("checked"),
                        "changed": changed,
                    },
                    severity="info",
                )
            except RuntimeBlocked as rb:
                record_event(
                    "runtime_guard_block",
                    {"reason": str(rb), "table": table_id},
                    severity="error",
                )
                # עצירת rollout; אם block_on_drift=True, ה־RuntimeBlocked כבר נזרק מתוך check_runtime_table
                raise RolloutBlocked(f"runtime_error: {rb}")
    return {
        "ok": True,
        "general": res_general,
        "tables_checked": len(table_specs),
        "runtime": {
            "sampled": runtime_sampled,
            "checked": runtime_checked,
            "tables": runtime_tables,
        },
    }