# imu_repo/tests/test_runtime_lineage.py
from __future__ import annotations
import os, json, tempfile, shutil
from engine.runtime_guard import check_runtime_table, RuntimeBlocked
from provenance.runtime_lineage import get_last

def _fetcher_rows(rows):
    payload = json.dumps({"items": rows}).encode("utf-8")
    def f(url: str) -> bytes:
        assert url.startswith("https://api.example.com")
        return payload
    return f

def test_lineage_records_and_no_drift_block():
    tmp = tempfile.mkdtemp(prefix="imu_test_")
    os.environ["IMU_HOME"] = tmp
    url = "https://api.example.com/orders"
    spec = {
        "path":"page.components[0]",
        "binding_url":url,
        "columns":[
            {"name":"id","type":"string","required":True},
            {"name":"n","type":"number"}
        ],
        "filters":None,"sort":None
    }
    rows1 = [{"id":"A","n":1},{"id":"B","n":2}]
    res1 = check_runtime_table(spec, policy={"runtime_check_enabled":True}, fetcher=_fetcher_rows(rows1))
    assert res1["ok"] and res1["hash"]

    # ריצה עם אותו תוכן — אין drift
    res2 = check_runtime_table(spec, policy={"runtime_check_enabled":True, "prev_content_hash":res1["hash"]},
                               fetcher=_fetcher_rows(rows1))
    assert res2["ok"] and res2["hash"] == res1["hash"]
    shutil.rmtree(tmp, ignore_errors=True)

def test_drift_block_when_enabled():
    tmp = tempfile.mkdtemp(prefix="imu_test_")
    os.environ["IMU_HOME"] = tmp
    url = "https://api.example.com/orders"
    spec = {
        "path":"page.components[0]",
        "binding_url":url,
        "columns":[
            {"name":"id","type":"string","required":True}
        ],
        "filters":None,"sort":None
    }
    rows1 = [{"id":"A"},{"id":"B"}]
    rows2 = [{"id":"A"},{"id":"C"},{"id":"D"}]  # שינוי תוכן -> hash חדש
    res1 = check_runtime_table(spec, policy={"runtime_check_enabled":True}, fetcher=_fetcher_rows(rows1))
    try:
        check_runtime_table(
            spec,
            policy={"runtime_check_enabled":True, "block_on_drift":True, "prev_content_hash":res1["hash"]},
            fetcher=_fetcher_rows(rows2)
        )
        assert False, "expected RuntimeBlocked due to drift"
    except RuntimeBlocked as e:
        assert "runtime_drift" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)