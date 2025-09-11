# imu_repo/ui/package.py
from __future__ import annotations
import json, hashlib, html, os
from typing import Dict, Any, Optional, List
from grounded.claims import current
from ui.dsl import Page
from ui.render import render_html
from security.signing import sign_manifest
from provenance.cas import CAS
from provenance.provenance import ProvenanceStore

def sha256_hex(data: bytes) -> str:
    h = hashlib.sha256(); h.update(data); return h.hexdigest()

def build_ui_artifact(page: Page, *, nonce: str="IMU_NONCE", key_id: str="default",
                      cas_root: Optional[str]=None, min_trust: float=0.75) -> Dict[str, Any]:
    html_text = render_html(page, nonce=nonce)
    digest = sha256_hex(html_text.encode("utf-8"))
    evs = current().snapshot()
    # ===== provenance (CAS) =====
    cas_info = None
    if cas_root:
        cas     = CAS(cas_root)
        store   = ProvenanceStore(cas, min_trust=min_trust)
        ev_sha  = store.ingest_evidences(evs)
        attach  = store.attach_artifact(html_text.encode("utf-8"),
                                        meta={"kind":"ui.html","mime":"text/html","title": page.title, "sha256_hex": digest},
                                        evidences_sha=ev_sha, key_id=key_id)
        cas_info = {"artifact_sha": attach["artifact_sha"], "manifest_sha": attach["manifest_sha"], "agg_trust": attach["agg_trust"]}

    kinds: Dict[str,int] = {}
    for e in evs: kinds[e["kind"]] = kinds.get(e["kind"],0)+1
    manifest = {
        "title": page.title,
        "sha256_hex": digest,
        "evidences": {"count": len(evs), "kinds": kinds},
        "meta": {"permissions": page.permissions},
        "provenance": cas_info or {}
    }
    signed = sign_manifest(manifest, key_id=key_id)

    def _esc(s: str) -> str: return html.escape(str(s), quote=True)
    rows = "\n".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k,v in kinds.items())
    prov_rows = ""
    if cas_info:
        prov_rows = f"<tr><td>artifact_sha</td><td><code>{_esc(cas_info['artifact_sha'])}</code></td></tr>" \
                    f"<tr><td>manifest_sha</td><td><code>{_esc(cas_info['manifest_sha'])}</code></td></tr>" \
                    f"<tr><td>agg_trust</td><td>{cas_info['agg_trust']:.2f}</td></tr>"
    report = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>IMU UI Report: {_esc(page.title)}</title>
<style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;padding:20px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:6px 8px}}
.badge{{display:inline-block;background:#eee;border-radius:10px;padding:2px 6px}}</style></head><body>
<h1>UI Artifact Report</h1>
<p>Title: <b>{_esc(page.title)}</b></p>
<p>SHA-256: <code>{_esc(digest)}</code></p>
<p>Signature alg/key_id: <code>{_esc(signed.get('signature',{{}}).get('alg','HMAC-SHA256'))}</code> /
   <code>{_esc(signed.get('signature',{{}}).get('key_id','default'))}</code></p>
<p>MAC: <code>{_esc(signed.get('signature',{{}}).get('mac',''))}</code></p>
<h3>Evidences</h3>
<p>Total: <span class="badge">{len(evs)}</span></p>
<table><thead><tr><th>Kind</th><th>Count</th></tr></thead><tbody>{rows}</tbody></table>
<h3>Provenance</h3>
<table><tbody>{prov_rows}</tbody></table>
</body></html>"""

    return {"html": html_text, "sha256": digest, "manifest": signed, "report_html": report, "provenance": cas_info or {}}