# imu_repo/engine/pipeline_multi.py
from __future__ import annotations
import os, json, time
from typing import Dict, Any, List
from synth.specs import BuildSpec
from engine.micro_split import split_spec
from pipeline.synthesis import SynthesisPipeline 
from kpi.aggregate import aggregate

SP = SynthesisPipeline

def run_pipeline_multi(spec: BuildSpec, out_root: str="/mnt/data/imu_builds", user_id: str="anon") -> Dict[str,Any]:
    """
    מפצל את ה-spec לשירותים, מריץ run_pipeline לכל שירות, אוגר תוצאות,
    מחשב KPI כולל + רול־אאוט כולל.
    """
    os.makedirs(out_root, exist_ok=True)
    comps = split_spec(spec)
    results: List[Dict[str,Any]] = []
    for s in comps:
        # המדיניות פר־שירות: app name = spec.name:role
        r = SP.run(s, out_root=out_root, user_id=user_id)
        results.append(r)

    agg = aggregate(results)
    bundle_dir = os.path.join(out_root, f"{int(time.time()*1000)}_{spec.name}_bundle")
    os.makedirs(bundle_dir, exist_ok=True)
    with open(os.path.join(bundle_dir, "multi_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"components": results, "aggregate": agg}, f, ensure_ascii=False, indent=2)
    return {"components": results, "aggregate": agg, "bundle_dir": bundle_dir}