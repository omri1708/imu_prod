# engine/pipeline_events.py (פאבליש לאירועים + אכיפת Contracts/Evidence)
# -*- coding: utf-8 -*-
import json, time, uuid, os
from typing import Dict, Any, List
from broker.stream import broker
from audit.log import AppendOnlyAudit
from governance.user_policy import Policy, EvidenceIndex
from engine.adapter_registry import get_adapter
from synth.specs_adapter import parse_adapter_jobs
from engine.contracts_gate import enforce_respond_contract
import time, random
from broker.stream import broker


AUDIT = AppendOnlyAudit("var/audit/pipeline.jsonl")


def emit_progress(pct: float):
    broker.publish("progress", {"ts": time.time(), "value": float(pct)}, priority="logic")


def emit_timeline(kind: str, msg: str):
    broker.publish("timeline", {"ts": time.time(), "kind": kind, "msg": msg}, priority="telemetry")


def _emit(topic: str, event: dict, *, priority: int = 1):
    ok = broker.publish(topic, event, priority=priority)
    AUDIT.append({"topic":topic, "delivered":ok, "event":event})

def run_pipeline_spec(*, user: str, spec_text: str, policy, ev_index) -> str:
    """
    מריץ pipeline לפי spec JSON:
    {
      "name":"..",
      "targets":[{"path":"...", "content":"..."}],
      "adapters":[{"kind":"k8s", "manifest":"..."}, {"kind":"android"}, ...]
    }
    """
    run_id = f"run-{int(time.time()*1000)}"
    ws = os.path.abspath(os.path.join("var", "runs", run_id))
    os.makedirs(ws, exist_ok=True)
    meta = {"run_id": run_id, "user": user}
    _emit("progress", {"stage":"init","run_id":run_id,"user":user}, priority=0)
    emit_timeline("run.start", f"{run_id} user={user}")
    # סימולציית שלבים בצנרת — מחליפים בחיבור שלך לצנרת אמיתית
    for step in range(0, 101, 5):
        emit_progress(step)
        if step in (10, 50, 90):
            emit_timeline("stage", f"stage at {step}%")
        time.sleep(0.02 + random.random()*0.01)
    emit_timeline("run.done", f"{run_id} ok")
    # כתיבת יעדים (קבצים) אם ניתנו
    import pathlib
    try:
        spec = json.loads(spec_text)
    except Exception as e:
        _emit("timeline", {"t":"spec_error","err":str(e), **meta}, priority=0)
        raise

    for t in spec.get("targets", []) or []:
        p = os.path.join(ws, t.get("path","artifact.txt"))
        pathlib.Path(os.path.dirname(p)).mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(t.get("content",""))
        _emit("timeline", {"t":"target_written","path":p, **meta})

    # אדפטורים
    jobs = parse_adapter_jobs(spec_text)
    if not jobs:
        _emit("timeline", {"t":"no_adapters","msg":"no adapter jobs in spec", **meta})
    artifacts: List[Dict[str,str]] = []
    total = len(jobs) or 1
    for i, job in enumerate(jobs, 1):
        kind = job["kind"]
        _emit("progress", {"stage":"adapter_start","kind":kind,"i":i,"n":total, **meta})
        try:
            ad = get_adapter(kind)
            res = ad.build(job, user, ws, policy, ev_index)
            # חוזה: עבור claims -> חייב evidence ב-CAS
            enforce_respond_contract(stage=f"adapter:{kind}", claims=res.claims, evidence=res.evidence, policy=policy, ev_index=ev_index)
            artifacts.append(res.artifacts)
            _emit("timeline", {"t":"adapter_ok","kind":kind,"claims":res.claims,"evidence":res.evidence, **meta})
        except Exception as e:
            _emit("timeline", {"t":"adapter_err","kind":kind,"err":str(e), **meta}, priority=0)
            raise
        finally:
            _emit("progress", {"stage":"adapter_done","kind":kind,"i":i,"n":total, **meta})

    _emit("progress", {"stage":"complete","run_id":run_id, **meta}, priority=0)
    return run_id