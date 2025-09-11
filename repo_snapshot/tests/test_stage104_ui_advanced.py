# imu_repo/tests/test_stage104_ui_advanced.py
from __future__ import annotations
import os, json
from typing import Dict, Any
from engine.policy_compiler import strict_prod_from
from engine.verifier_km import as_quorum_member_with_policy
from db.strict_repo import StrictRepo
from ui_dsl.renderer_v2 import AdvancedStrictUIRenderer
from cas.store import stat

def _base_policy() -> Dict[str,Any]:
    base = {
        "trust_domains": {"example.com":5},
        "trusted_domains": ["example.com"],
        "signing_keys": {"root":{"secret_hex":"aa"*32,"algo":"sha256"}},
        "min_distinct_sources": 1,
        "min_total_trust": 1,
        "min_provenance_level": 1,
        "default_freshness_sec": 1200,
        "perf_sla": {
            "latency_ms": {"p95_max": 150.0},
            "throughput_rps": {"min": 50.0},
            "error_rate": {"max": 0.10},
            "near_miss_factor": 1.20
        },
        "consistency": {
            "drift_pct": 0.15,
            "near_miss_streak_heal_threshold": 3,
            "heal_action": "freeze_autotune"
        }
    }
    return strict_prod_from(json.dumps(base))

def test_advanced_grid_and_table(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    os.environ["IMU_STATE_DIR"] = str(tmp_path / ".state")
    os.environ["IMU_CAS_DIR"] = str(tmp_path / ".cas")

    pol = _base_policy()
    v  = as_quorum_member_with_policy(pol, expected_scope="deploy")

    repo = StrictRepo(path=None)
    rows, claims = repo.query("SELECT id,name,score FROM sample ORDER BY id ASC")

    # UI מתקדם: גריד עם אזורים + טבלה עם freeze/sort/filters
    ui_spec = {
        "type":"grid",
        "rows":"auto 1fr auto",
        "cols":"220px 1fr",
        "areas":[
            "sidebar content",
            "sidebar content",
            "footer  footer"
        ],
        "gap":"10px",
        "children":{
            "sidebar":"<div><h4>Menu</h4><ul><li>A</li><li>B</li></ul></div>",
            "content":{
                "type":"table",
                "columns":[
                    {"field":"id","title":"ID","width":"80"},
                    {"field":"name","title":"Name","width":"160"},
                    {"field":"score","title":"Score","width":"120"}
                ],
                "freeze": 2,
                "search": True
            },
            "footer":"<small>© demo</small>"
        }
    }

    rnd = AdvancedStrictUIRenderer(base_policy=pol)
    ctx = {"user":{"tier":"standard"}}
    out = rnd.render_and_package(ctx=ctx, ui_spec=ui_spec, data_provider=lambda _ctx: (rows, claims))

    assert out["ok"] and isinstance(out["bundle"], dict)
    b = out["bundle"]

    # ודא שקיים גם ui_provenance וגם ui_version
    types = [c.get("type") for c in (b.get("claims") or [])]
    assert "ui_provenance" in types, "missing ui_provenance"
    assert "ui_version" in types, "missing ui_version"

    # בדיקה שהמניפסטים קיימים ב-CAS
    ui_manifest_sha = [c["value"] for c in b["claims"] if c["type"]=="ui_provenance"][0]
    ver_sha = [c["value"] for c in b["claims"] if c["type"]=="ui_version"][0]
    assert stat(ui_manifest_sha) is not None
    assert len(ver_sha) == 64  # sha256

    # הפלט הטקסטואלי חייב לכלול מאפייני freeze (data-freeze) וקליינט סייד סקריפט
    assert "data-freeze" in out["text"]
    assert "data-widget='adv-table'" in out["text"]
    assert "<script>" in out["text"]