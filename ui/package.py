# imu_repo/ui/package.py
from __future__ import annotations
import json, hashlib, html
from typing import Dict, Any
from grounded.claims import current
from ui.dsl import Page
from ui.render import render_html
from security.signing import sign_manifest

def sha256_hex(data: bytes) -> str:
    h = hashlib.sha256(); h.update(data); return h.hexdigest()

def build_ui_artifact(page: Page, *, nonce: str="IMU_NONCE", key_id: str="default") -> Dict[str, Any]:
    html_text = render_html(page, nonce=nonce)
    digest = sha256_hex(html_text.encode("utf-8"))
    evs = current().snapshot()
    kinds: Dict[str,int] = {}
    for e in evs: kinds[e["kind"]] = kinds.get(e["kind"],0)+1
    manifest = {
        "title": page.title,
        "sha256_hex": digest,
        "evidences": {"count": len(evs), "kinds": kinds},
        "meta": {"permissions": page.permissions}
    }
    signed = sign_manifest(manifest, key_id=key_id)

    def _esc(s: str) -> str: return html.escape(str(s), quote=True)
    rows = "\n".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k,v in kinds.items())
    report = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>IMU UI Report: {_esc(page.title)}</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;padding:20px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:6px 8px}}
.badge{{display:inline-block;background:#eee;border-radius:10px;padding:2px 6px}}</style></head><body>
<h1>UI Artifact Report</h1>
<p>Title: <b>{_esc(page.title)}</b></p>
<p>SHA-256: <code>{_esc(digest)}</code></p>
<p>Signature alg/key_id: <code>{_esc(signed['signature']['alg'])}</code> / <code>{_esc(signed['signature']['key_id'])}</code></p>
<p>MAC: <code>{_esc(signed['signature']['mac'])}</code></p>
<h3>Evidences</h3>
<p>Total: <span class="badge">{len(evs)}</span></p>
<table><thead><tr><th>Kind</th><th>Count</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""

    return {"html": html_text, "sha256": digest, "manifest": signed, "report_html": report}