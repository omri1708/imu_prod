# imu_repo/tests/test_stage103_end2end_ui_realtime_db.py
from __future__ import annotations
import os, json, time
from typing import Dict, Any
from engine.policy_compiler import strict_prod_from
from engine.verifier_km import as_quorum_member_with_policy
from ui_dsl.strict_renderer import StrictUIRenderer
from db.strict_repo import StrictRepo
from realtime.strict_ws import StrictWSMux
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

def test_ui_db_realtime_strict(tmp_path):
    os.environ["IMU_AUDIT_DIR"] = str(tmp_path / ".audit")
    os.environ["IMU_STATE_DIR"] = str(tmp_path / ".state")
    os.environ["IMU_CAS_DIR"] = str(tmp_path / ".cas")

    pol = _base_policy()
    v = as_quorum_member_with_policy(pol, expected_scope="deploy")

    # DB
    repo = StrictRepo(path=None)
    rows, claims = repo.query("SELECT id,name,score FROM sample ORDER BY id ASC")

    # UI spec (טבלה בסיסית)
    ui_spec = {
        "type": "table",
        "columns": [{"field":"id","title":"ID"},{"field":"name","title":"Name"},{"field":"score","title":"Score"}]
    }

    # Render+Package
    rnd = StrictUIRenderer(base_policy=pol)
    ctx = {"user":{"tier":"standard"}}
    out = rnd.render_and_package(ctx=ctx, ui_spec=ui_spec, data_provider=lambda _ctx: (rows, claims))
    assert out["ok"] and isinstance(out["bundle"], dict)
    # ודא שקיים Claim של ui_provenance ומניפסט נשמר ב־CAS
    cl = out["bundle"].get("claims") or []
    ui_cl = [c for c in cl if c.get("type")=="ui_provenance"]
    assert ui_cl, "missing ui_provenance claim"
    msha = ui_cl[0]["value"]
    assert stat(msha) is not None, "ui manifest not in CAS"

    # Realtime strict
    mux = StrictWSMux(base_policy=pol)
    ev = mux.send(ctx=ctx, channel="events", payload={"kind":"refresh","count":len(rows)}, claims=claims)
    assert ev["ok"] and isinstance(ev["bundle"], dict)
    # חבילת רילטיים גם מכילה claims
    assert (ev["bundle"].get("claims") or []), "realtime bundle must include claims"