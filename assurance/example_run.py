# -*- coding: utf-8 -*-
"""
דוגמת ריצה שמראה:
1) הוספת קלטים (bytes/file) ל-CAS
2) הוספת Claim/Evidence/Validator
3) Builder שמייצר ארטיפקט
4) Gate: בלי evidence/validators → commit נכשל; איתם → commit חתום
"""
from __future__ import annotations
from assurance.assurance import AssuranceKernel
from assurance.validators import schema_validator, unit_range_validator
from assurance.ops_registry import text_render_builder, shell_builder
from assurance.signing import Signer
from assurance.errors import RefusedNotGrounded, ValidationFailed, ResourceRequired
import os, json, tempfile, shutil

def main():
    root = "./assurance_store_demo"
    if os.path.exists(root):
        shutil.rmtree(root)
    kernel = AssuranceKernel(root)

    # --- התחלת סשן ---
    s = kernel.begin("report.generate", "1.0.0", "Generate JSON report with measured value")

    # inputs
    s.add_input_bytes("spec", b'{"title":"demo","value":42}')
    # claims (טענות שמתחייבות להיבדק)
    s.add_claim("title", "demo")
    s.add_claim("value", 42, units="ms", bounds={"min": 0, "max": 100})

    # evidence (במציאות: מקור חתום/URL/קובץ + digest; כאן פשוט מציינים מקור)
    s.add_evidence(kind="measurement", source="internal:test-sensor", trust=0.8)

    # validators: סכימה + טווח יחידות
    schema = {
        "required": ["title", "value"],
        "properties": {"title": {"type": "string", "minLength": 1},
                       "value": {"type": "number", "minimum": 0, "maximum": 100}}
    }
    s.attach_validator(schema_validator(schema))
    s.attach_validator(unit_range_validator("value", "ms", minimum=0, maximum=100))

    # builder #1: רינדור טקסט → JSON; אפשר גם shell_builder(["echo","..."])
    payload = {"title": "demo", "value": 42}
    s.set_builder(text_render_builder(json.dumps(payload, ensure_ascii=False)))

    # build
    out_digests = s.build()
    print("build outputs:", out_digests)

    # validate+commit (עם חתימה HMAC כברירת מחדל; אפשר לספק Ed25519 PEM)
    signer = Signer(key_id="demo-hmac")
    record = s.commit(signer)
    print("commit:", record["root"], record["signature"])

    # הדגמת resource_required: נסה להריץ כלי חסר
    try:
        s2 = kernel.begin("shell.echo", "1.0.0", "echo hello world")
        s2.add_evidence("source", "internal:test", trust=0.9)
        s2.attach_validator(schema_validator({"required": [], "properties": {}}))
        s2.set_builder(shell_builder(["nonexistent_tool_xyz", "arg"]))
        s2.build()
        s2.commit(signer)
    except ResourceRequired as e:
        print("as expected:", str(e))

if __name__ == "__main__":
    main()
