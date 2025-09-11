from __future__ import annotations
from engine.cas_store import put_json, get_json
from engine.json_diff import diff_paths

def test_cas_put_get_idempotent():
    obj = {"a": 1, "b": {"c": [1,2,3]}}
    h1 = put_json(obj)
    h2 = put_json(obj)  # אמור להיות אותו hash
    assert h1 == h2
    back = get_json(h1)
    assert back == obj

def test_json_diff_various():
    a = {"x": 1, "y": {"z": [1,2,3]}}
    b = {"x": 1, "y": {"z": [1,3,4], "w": 5}}
    paths = diff_paths(a, b)
    # בודקים שנרשמו לפחות המסלולים המשמעותיים
    assert any(p.endswith(".y.z[1]") for p in paths)  # 2 → 3
    assert any(p.endswith(".y.z[2]") for p in paths)  # 3 → 4
    assert any(p.endswith(".y.w") for p in paths)     # key חדש
