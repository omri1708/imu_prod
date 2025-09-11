# imu_repo/tests/external_validation.py
from __future__ import annotations
import os, json, time
from engine.pipeline import Engine

def prepare_fixture():
    root = ".imu_state/httpcache/example.gov"
    os.makedirs(root, exist_ok=True)
    doc = {"title":"Gov Open Data Catalog", "version":"2025-09-01", "age_sec": 60}
    with open(os.path.join(root, "catalog.json"),"w",encoding="utf-8") as f:
        json.dump(doc,f)

def program_ok():
    return [
        {"op":"PUSH","value":"Gov Open Data Catalog"},{"op":"STORE","reg":"title"},
        {"op":"PUSH","value":"2025-09-01"},{"op":"STORE","reg":"version"},
        {"op":"PUSH","value":60},{"op":"STORE","reg":"age_sec"},
        {"op":"EVIDENCE","claim":"http_doc","sources":["httpcache://example.gov/catalog.json"]},
        {"op":"RESPOND","status":200,"body":{"title":"reg:title","version":"reg:version","age_sec":"reg:age_sec"}}
    ]

def program_stale():
    return [
        {"op":"PUSH","value":"Old Doc"},{"op":"STORE","reg":"title"},
        {"op":"PUSH","value":"2020-01-01"},{"op":"STORE","reg":"version"},
        {"op":"PUSH","value":99999999},{"op":"STORE","reg":"age_sec"},
        {"op":"EVIDENCE","claim":"http_doc","sources":["httpcache://example.gov/catalog.json"]},
        {"op":"RESPOND","status":200,"body":{"title":"reg:title","version":"reg:version","age_sec":"reg:age_sec"}}
    ]

def run():
    prepare_fixture()
    e=Engine(mode="strict")
    c1,b1 = e.run_program(program_ok(), {}, policy="strict")
    print("ok:", c1, b1)
    # נעדכן את המדיניות להגבלת max_age ל-120 שניות (זה default כבר), ונריץ תוכנית ש"שקרית": age_sec גבוה מדי → הוולידטור יפיל
    c2,b2 = e.run_program(program_stale(), {}, policy="strict")
    print("stale:", c2, b2)
    # הצלחה אם ok==200 ו-stale==412
    return 0 if (c1==200 and c2==412) else 1

if __name__=="__main__":
    raise SystemExit(run())