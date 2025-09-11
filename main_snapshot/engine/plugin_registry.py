# imu_repo/engine/plugin_registry.py
from __future__ import annotations
from typing import Dict, Any, List
import os, json

from plugins.db.sqlite_sandbox import SQLiteSandbox
from plugins.ui.static_site import StaticSite
from plugins.compute.vector_ops import VectorOps

PLUGINS = {
    "db/sqlite": SQLiteSandbox,
    "ui/static": StaticSite,
    "compute/vector": VectorOps,
}

def run_plugins(spec: Any, build_dir: str, user_id: str) -> Dict[str,Any]:
    extras = getattr(spec, "extras", {}) or {}
    requested: Dict[str, dict] = {}
    # convention: extras["plugins"] = {"db/sqlite": {...}, "ui/static": {...}, ...}
    requested = extras.get("plugins") or {}
    out = {"evidence":{}, "kpi_parts":[], "plugins":[]}

    for name, cfg in requested.items():
        if name not in PLUGINS:
            raise RuntimeError(f"unknown_plugin:{name}")
        cls = PLUGINS[name]
        inst = cls(**(cfg or {})) if isinstance(cfg, dict) else cls()
        res = inst.run(spec, build_dir, user_id)
        out["plugins"].append(name)
        out["evidence"][name] = res.get("evidence", {})
        k = res.get("kpi", {})
        if k: out["kpi_parts"].append({"name": name, **k})
    # KPI מצרפי פשוט של התוספים
    if out["kpi_parts"]:
        score = sum(p.get("score",0.0) for p in out["kpi_parts"]) / max(1, len(out["kpi_parts"]))
        out["kpi"] = {"score": score}
    else:
        out["kpi"] = {"score": 0.0}
    # כתיבה לקובץ ראיות
    with open(os.path.join(build_dir, "plugin_evidence.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out