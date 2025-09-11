# imu_repo/tests/test_stage49_plugins.py
from __future__ import annotations
from synth.specs import BuildSpec, Contract
from engine.synthesis_pipeline import run_pipeline
from grounded.source_policy import policy_singleton as SourcePolicy

def _schema():
    return {
        "type":"object",
        "properties":{
            "tests":{"type":"object"},
            "perf":{"type":"object","properties":{"p95_ms":{"type":"number","maximum":1500}},"required":["p95_ms"]},
            "ui":{"type":"object","properties":{"score":{"type":"number","minimum":70}},"required":["score"]},
            "plugins":{"type":"object"}
        },
        "required":["tests","perf","ui"]
    }

def run():
    SourcePolicy.set_allowlist(["internal.test"])
    spec = BuildSpec(
        name="stage49_plugins",
        kind="web_service",
        language_pref=["python"],
        ports=[19595],
        endpoints={"/hello":"hello_json","/ui":"static_ui"},
        contracts=[Contract(name="svc", schema=_schema())],
        evidence_requirements=["service_tests","perf_summary","ui_accessibility","plugin_evidence"],
    )
    # מוסיפים extras (לא מחייב שינוי מחלקה – getattr ב-run_plugins יתפוס dict)
    setattr(spec, "extras", {
        "plugins": {
            "db/sqlite": {
                "max_rows": 1000,
                "max_ms": 1200
            },
            "ui/static": {},
            "compute/vector": {"max_len": 1500}
        },
        "db": {
            "schema": [
                "CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, age INT)"
            ],
            "seed": [
                ("INSERT INTO users(name,age) VALUES(?,?)", ["Noa",29]),
                ("INSERT INTO users(name,age) VALUES(?,?)", ["Avi",34]),
                ("INSERT INTO users(name,age) VALUES(?,?)", ["Maya",26])
            ],
            "queries": [
                "SELECT name, age FROM users WHERE age>=28 ORDER BY age DESC"
            ]
        },
        "ui": {
            "pages":[
                {"path":"/index.html","title":"App","body":"<h1>Welcome</h1><p>Hello!</p>"}
            ]
        },
        "compute": {
            "n": 800
        }
    })

    s = run_pipeline(spec, user_id="u49")
    # בדיקות: יש plugin_evidence, rollout מאושר, ו־KPI כולל סביר
    ok = (
        s["rollout"]["approved"] and
        "plugin_evidence" in s["evidence"] and
        s["kpi"]["score"] >= 70.0 and
        "db/sqlite" in s["evidence"]["plugin_evidence"] and
        "ui/static" in s["evidence"]["plugin_evidence"] and
        "compute/vector" in s["evidence"]["plugin_evidence"]
    )
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())