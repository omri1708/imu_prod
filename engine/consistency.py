# imu_repo/engine/consistency.py
from __future__ import annotations
from typing import Any, Dict, List, DefaultDict
from collections import defaultdict
from synth.schema_validate import ClaimSchemaError, validate_claim_schema, consistent_numbers

class ConsistencyError(Exception): ...

def validate_claims_and_consistency(
    claims: List[Dict[str,Any]],
    *,
    require_consistency_groups: bool,
    default_number_tolerance: float
) -> None:
    """
    1) ולידציה של סכימות/טווחים לכל claim (אם קיימת schema).
    2) הצלבה בתוך קבוצות consistency_group: כל הערכים המספריים צריכים להיות עקביים
       עד כדי tolerance (מתוך הסכימה או ברירת מחדל).
    """
    groups: DefaultDict[str, List[Dict[str,Any]]] = defaultdict(list)
    for c in claims:
        validate_claim_schema(c)
        grp = c.get("consistency_group")
        if grp:
            groups[str(grp)].append(c)

    if require_consistency_groups:
        for g, arr in groups.items():
            # מחפשים value+schema type=number
            values = []
            tol = None
            for c in arr:
                sch = c.get("schema") or {}
                if sch.get("type") == "number" and "value" in c:
                    v = float(c["value"])
                    values.append(v)
                    if tol is None:
                        tol = float(sch.get("tolerance", default_number_tolerance))
            if len(values) >= 2:
                t = float(tol if tol is not None else default_number_tolerance)
                base = values[0]
                for v in values[1:]:
                    if not consistent_numbers(base, v, t):
                        raise ConsistencyError(f"inconsistent values in group '{g}': {values} with tol={t}")