from __future__ import annotations
import json, os
from pathlib import Path
from engine import config as cfgmod

def test_runtime_json_loader_reads_policy(tmp_path, monkeypatch):
    root = tmp_path / "imu_repo"
    conf = root / "config"; conf.mkdir(parents=True)
    (root / "snapshots").mkdir(parents=True)

    runtime = {
        "composer": {"mode": "merge"},
        "policy": {
            "runtime_check_enabled": True,
            "runtime_prev_hash_map": {"https://api.example.com/orders": "9e1c...a2f"}
        }
    }
    (conf / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")

    # מכוונים את ה-loader לשורש חדש
    monkeypatch.setattr(cfgmod, "ROOT", str(root))
    monkeypatch.setattr(cfgmod, "CFG_DIR", str(conf))
    monkeypatch.setattr(cfgmod, "CFG_FILE", str(conf / "runtime.json"))

    cfg = cfgmod.load_config()
    assert (cfg.get("composer") or {}).get("mode") == "merge"
    assert cfg.get("policy", {}).get("runtime_prev_hash_map")
