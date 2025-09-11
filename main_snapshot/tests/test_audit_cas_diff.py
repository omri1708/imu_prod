from __future__ import annotations
import json
from pathlib import Path

from engine.auto_remediation import apply_remedies, Remedy
from engine.audit_log import _audit_root

def test_apply_remedies_writes_audit_log_and_cas(tmp_path):
    # נבנה שינוי קטן: הסרת פילטר + שינוי policy
    policy = {
        "allow_remove_filter_if_blocked": True,
        "allow_update_prev_hash_on_schema_ok": True,
        "runtime_prev_hash_map": {}
    }
    specs = [{
        "path": "page.components[0]",
        "binding_url": "https://api.example.com/orders",
        "columns": [{"name": "order_id", "type": "string", "required": True}],
        "filters": {"amount": {"op": ">=", "value": 100}},
        "sort": None
    }]

    def _drop_filter(pol, ts):
        ts[0]["filters"] = {}

    def _accept_hash(pol, ts):
        pol["runtime_prev_hash_map"]["https://api.example.com/orders"] = "abcd"*16

    remedies = [
        Remedy("Remove blocking filter", "conservative", _drop_filter),
        Remedy("Accept new hash", "conservative", _accept_hash),
    ]

    apply_remedies(remedies, policy=policy, table_specs=specs)

    # בדיקת audit.log.jsonl + CAS
    audit_dir = Path(_audit_root())
    log = audit_dir / "audit.log.jsonl"
    assert log.exists(), "audit log not written"

    last = list(log.read_text(encoding="utf-8").splitlines())[-1]
    entry = json.loads(last)
    # יש חתימה
    assert "signature" in entry and entry["signature"]["mode"] in ("hmac-sha256","sha256")
    # יש CAS before/after
    assert entry.get("cas") and entry["cas"].get("before") and entry["cas"].get("after")
    # יש רשימת נתיבים שהשתנו
    changed = entry.get("changed_paths") or []
    # נוודא שהשינוי נוגע לפילטרים או למפה
    joined = ".".join(changed)
    assert ("table_specs" in joined) or ("runtime_prev_hash_map" in joined)
