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

    return {
        "ok": True,
        "general": res_general,
        "tables_checked": len(table_specs)
    }