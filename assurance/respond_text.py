# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from typing import Dict, Any, List
from assurance.assurance import AssuranceKernel
from assurance.signing import Signer
from assurance.validators import schema_validator
from assurance.errors import RefusedNotGrounded, ResourceRequired, ValidationFailed
from grounding.providers import FileProvider, HTTPProvider, add_evidence_to_session

class GroundedResponder:
    """
    מייצר תשובה טקסטואלית *רק* אם יש Evidence + Validators עוברים; אחרת מסרב/דורש משאב.
    """
    def __init__(self, store_root: str = "./assurance_store_text"):
        self.kernel = AssuranceKernel(store_root)

    def respond_from_sources(self, prompt: str, sources: List[Dict[str,Any]]) -> Dict[str,Any]:
        """
        sources: [{ "file": "path.txt" } | { "url": "https://..." } ...]
        """
        sess = self.kernel.begin("respond.text", "1.0.0", f"Grounded response: {prompt[:80]}")
        # claims: תשובה חייבת לכלול ציון מקורות
        sess.add_claim("has_citations", True)
        # validator סמלי: JSON schema לתשובה (text + citations)
        schema = {
            "required": ["text", "citations"],
            "properties": {
                "text": {"type":"string", "minLength": 1},
                "citations": {"type":"array"}
            }
        }
        sess.attach_validator(schema_validator(schema))

        fp = FileProvider(self.kernel.cas)
        try:
            hp = HTTPProvider(self.kernel.cas)
        except Exception:
            hp = None  # אם אין requests נחיה בלי HTTP

        digests = []
        for s in sources:
            if "file" in s:
                ev = fp.fetch(s["file"], trust=0.7)
                add_evidence_to_session(sess, ev)
                digests.append((ev.source, ev.digest))
            elif "url" in s:
                if not hp: raise ResourceRequired("tool:requests", "pip install requests")
                ev = hp.fetch(s["url"], trust=0.6)
                add_evidence_to_session(sess, ev)
                digests.append((ev.source, ev.digest))
            else:
                raise ValidationFailed("bad source item")

        if not digests:
            raise RefusedNotGrounded("no sources")

        # בניית תשובה פשוטה: טקסט + ציטוטים (מזהים/URLs) — אין “ניחוש”
        text = f"נענָה על סמך {len(digests)} מקורות. ראו ציטוטים."
        payload = {"text": text, "citations": [{"source": src, "digest": d} for (src,d) in digests]}
        sess.set_builder(lambda s: [s.cas.put_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                                    meta={"type":"grounded-text"})])

        outs = sess.build()
        rec = sess.commit(Signer("text-hmac"))
        return {"ok": True, "root": rec["root"], "manifest": rec["manifest_digest"], "payload": payload}
