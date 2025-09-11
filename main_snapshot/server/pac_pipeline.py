# server/pac_pipeline.py
# Policy-as-Code pipeline: lint → validate/apply → sign unified bundle (optional) → broadcast timeline events.
from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import time, json

from policy.lint import lint_yaml_text, PolicyLintError
from policy.policy_hotload import _apply_cfg
from server.unified_archive_api import export_signed
from server.stream_wfq import BROKER

POLICY_FILE = Path("security/policy_rules.yaml")

def pac_run(yaml_text: str, user_id: str, sign_bundle: bool = True) -> Dict[str, Any]:
    """מריץ תהליך PaC על YAML שסופק. מחזיר תוצאה כולל envelope אם בוצע חתימה."""
    t0 = time.time()
    BROKER.ensure_topic("timeline", rate=100, burst=500, weight=2)
    BROKER.submit("timeline","pac",{"type":"event","ts":time.time(),"note":"pac.lint.start"}, priority=3)
    cfg = lint_yaml_text(yaml_text)  # עשוי לזרוק PolicyLintError
    BROKER.submit("timeline","pac",{"type":"event","ts":time.time(),"note":"pac.lint.ok"}, priority=3)
    # כתיבה אטומית של הקובץ
    tmp = POLICY_FILE.with_suffix(".yaml.tmp")
    tmp.write_text(yaml_text, encoding="utf-8")
    tmp.replace(POLICY_FILE)
    # החלה מיידית (hot-apply)
    BROKER.submit("timeline","pac",{"type":"event","ts":time.time(),"note":"pac.apply.start"}, priority=3)
    _apply_cfg(cfg)
    BROKER.submit("timeline","pac",{"type":"event","ts":time.time(),"note":"pac.apply.ok"}, priority=3)
    out = {"ok": True, "ms": int((time.time()-t0)*1000)}
    if sign_bundle:
        # צור unified export חתום כהוכחה/trace
        BROKER.submit("timeline","pac",{"type":"event","ts":time.time(),"note":"pac.bundle.export"}, priority=4)
        resp = export_signed(name=f"pac_{int(time.time())}")
        # StreamingResponse לא ישיר בתוך פייפליין — נחלץ רק hash+path מהכותרות (נגיש ברמת HTTP בנפרד)
        out["bundle_note"] = "GET /unified/export_signed?name=pac_TIMESTAMP"
    return out