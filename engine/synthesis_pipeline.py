# imu_repo/engine/synthesis_pipeline.py
from __future__ import annotations
import json, hashlib, time
from typing import Dict, Any, Optional

from grounded.claims import current
from engine.capability_wrap import guard_text_capability_for_user
from synth.validators import (
    validate_spec, validate_plan, validate_package
)
from synth.generate_ab import generate_variants
from synth.generate_ab_prior import generate_variants_with_prior
from synth.generate_ab_explore import generate_variants_with_prior_and_explore
from engine.ab_selector import select_best
from engine.learn import learn_from_pipeline_result
from engine.learn_store import load_baseline, _task_key, load_history
from engine.config import load_config
from engine.explore_policy import decide_explore

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
    steps = [
        {"name":"generate_ab[prior/explore_fallback]"},
        {"name":"ab_select"},
        {"name":"package"},
    ]
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

async def run_pipeline(spec: Dict[str,Any], *, user_id: str, learn: bool = False) -> Dict[str,Any]:
    p = plan(spec)
    cfg = load_config()
    epsilon = float(cfg.get("explore", {}).get("epsilon", 0.0))

    key = _task_key(str(spec["name"]), str(spec["goal"]))
    baseline = load_baseline(key)

    if baseline is not None:
        hist = load_history(key, limit=500)
        want_explore = decide_explore(len(hist), epsilon)
        if want_explore:
            variants = generate_variants_with_prior_and_explore(spec, baseline)
            current().add_evidence("generate_ab_prior_explore",{
                "source_url":"local://generate_ab_prior_explore",
                "trust":0.95,
                "ttl_s":900,
                "payload":{
                    "labels":[v["label"] for v in variants],
                    "epsilon": epsilon,
                    "history_len": len(hist),
                    "baseline_summary":{
                        "label": baseline.get("label"),
                        "phi": float(baseline.get("phi", float('inf'))),
                        "p95_ms": float(baseline.get("p95_ms", float('inf'))),
                        "error_rate": float(baseline.get("error_rate", 1.0))
                    }
                }
            })
        else:
            variants = generate_variants_with_prior(spec, baseline)
            current().add_evidence("generate_ab_prior",{
                "source_url":"local://generate_ab_prior",
                "trust":0.95,
                "ttl_s":900,
                "payload":{"labels":[v["label"] for v in variants]}
            })
    else:
        variants = generate_variants(spec)
        current().add_evidence("generate_ab",{
            "source_url":"local://generate_ab",
            "trust":0.95,
            "ttl_s":900,
            "payload":{"count": len(variants), "labels":[v["label"] for v in variants]}
        })

    # ★ כאן ההתאמה האישית – A/B לפי Φ מרובה־יעדים עם משקולות מהפרופיל:
    winner = select_best(variants, spec=spec, user_id=user_id)

    pkg = _package_text(spec, winner)

    async def _emit_text(_: Dict[str,Any]) -> str:
        return pkg["artifact_text"]
    guarded = await guard_text_capability_for_user(_emit_text, user_id=user_id)
    out = await guarded({"ok": True})

    if learn:
        learn_from_pipeline_result(spec, winner, user_id=user_id)

    return out