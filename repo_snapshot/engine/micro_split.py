# imu_repo/engine/micro_split.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from copy import deepcopy
from synth.specs import BuildSpec, Contract

def _partition_endpoints(endpoints: Dict[str,str]) -> Tuple[Dict[str,str], Dict[str,str]]:
    """
    מפצל נקודות קצה לשני דומיינים:
      - api: כל מה שלא מתחיל ב-/bg
      - worker: כל מה שמתחיל ב-/bg (עבודות רקע)
    אם אין /bg כלל – נחלק חצי-חצי בקירוב יציב.
    """
    api, worker = {}, {}
    for k,v in (endpoints or {}).items():
        if k.startswith("/bg"):
            worker[k]=v
        else:
            api[k]=v
    if not worker and len(api) > 1:
        # פיצול יציב: חצי ראשון api, חצי שני worker
        items = list(api.items())
        mid = len(items)//2
        api = dict(items[:mid] or items[:1])
        worker = dict(items[mid:] or items[-1:])
    if not api and worker:
        # שיהיה שרת api בסיסי (בריאות/UI)
        api = {"/health":"health", "/ui":"static_ui"}
    if "/health" not in api:
        api["/health"] = "health"
    if "/ui" not in api:
        api["/ui"] = "static_ui"
    return api, worker

def derive_subspec(spec: BuildSpec, name_suffix: str, endpoints: Dict[str,str], port_base: int) -> BuildSpec:
    """
    גוזר BuildSpec לתת־שירות:
      - שם ייחודי name:suffix
      - פורט יחיד (base)
      - חוזים/עדויות נשמרים (ניתן לצמצם/להרחיב אם רוצים)
    """
    # חשוב: לא משנים את BuildSpec המקורי — יוצרים אחד חדש עם אותן שדות ידועים
    return BuildSpec(
        name=f"{spec.name}:{name_suffix}",
        kind=spec.kind,
        language_pref=list(spec.language_pref or []),
        ports=[port_base],
        endpoints=endpoints,
        contracts=list(spec.contracts or []),
        evidence_requirements=list(spec.evidence_requirements or []),
        external_evidence=list(spec.external_evidence or [])
    )

def split_spec(spec: BuildSpec) -> List[BuildSpec]:
    """
    מפצל spec לשני מיקרו־שירותים: api & worker.
    אם אין מה לפצל — מחזיר רשימה עם spec יחיד ששמו מסומן api.
    """
    eps = dict(spec.endpoints or {})
    api_eps, worker_eps = _partition_endpoints(eps)
    base_port = int((spec.ports or [18080])[0])

    subs: List[BuildSpec] = []
    if api_eps:
        subs.append(derive_subspec(spec, "api", api_eps, base_port))
    if worker_eps:
        subs.append(derive_subspec(spec, "worker", worker_eps, base_port+1))
    if not subs:
        # מינימום שירות api קטן
        subs.append(derive_subspec(spec, "api", {"/health":"health","/ui":"static_ui"}, base_port))
    return subs