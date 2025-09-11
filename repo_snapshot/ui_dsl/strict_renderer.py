# imu_repo/ui_dsl/strict_renderer.py
from __future__ import annotations
import json, html
from typing import Dict, Any, List, Callable, Tuple
from engine.respond_strict import RespondStrict
from ui_dsl.provenance import build_ui_provenance

DataProvider = Callable[[Dict[str,Any]], Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]]
# חוזה: data_provider(ctx) → (rows, claims)  ; claims = evidences על הנתונים

class StrictUIRenderer:
    """
    מרנדר UI-DSL ל־HTML+JS, ומכפיף את הפלט ל־Grounded-Strict:
      1) חייב claims על הנתונים.
      2) יוצר ui_provenance manifest ומוסיף אותו ל־claims.
      3) אורז את התגובה (טקסט HTML) בתוך proof חתום.
    """
    def __init__(self, *, base_policy: Dict[str,Any], http_fetcher=None, sign_key_id: str="root"):
        self.responder = RespondStrict(base_policy=base_policy, http_fetcher=http_fetcher, sign_key_id=sign_key_id)

    def _render_table(self, spec: Dict[str,Any], rows: List[Dict[str,Any]]) -> str:
        cols = spec.get("columns") or []
        thead = "".join(f"<th>{html.escape(c.get('title') or c.get('field') or '')}</th>" for c in cols)
        body_rows = []
        for r in rows:
            tds = []
            for c in cols:
                field = c.get("field")
                val = r.get(field, "")
                tds.append(f"<td>{html.escape(str(val))}</td>")
            body_rows.append("<tr>" + "".join(tds) + "</tr>")
        table = f"<table data-ui='table'><thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        return table

    def _render(self, ui_spec: Dict[str,Any], rows: List[Dict[str,Any]]) -> str:
        kind = ui_spec.get("type", "table")
        if kind == "table":
            return self._render_table(ui_spec, rows)
        # ניתן להרחיב לרכיבים נוספים (grid, chart ...) — כאן נעמוד בדרישה לטבלה
        return "<div>unsupported ui component</div>"

    def render_and_package(self,
                           *,
                           ctx: Dict[str,Any],
                           ui_spec: Dict[str,Any],
                           data_provider: DataProvider) -> Dict[str,Any]:
        rows, data_claims = data_provider(ctx)
        # Provenance ל־UI עצמו נכנס כ־claim נוסף
        sources = []
        for c in data_claims:
            for ev in c.get("evidence", []):
                if isinstance(ev, dict): sources.append(ev)
        prov = build_ui_provenance(ui_spec=ui_spec, sources=sources, policy=self.responder.base_policy)
        ui_claim = {
            "id": f"ui:{prov['manifest_sha256'][:16]}",
            "type": "ui_provenance",
            "text": "ui spec & assets integrity",
            "schema": {"type":"manifest","unit":""},
            "value": prov["manifest_sha256"],
            "evidence": [{
                "kind": "cas_manifest",
                "sha256": prov["manifest_sha256"]
            }],
            "consistency_group": "ui"
        }
        claims = list(data_claims) + [ui_claim]

        def _gen(_ctx: Dict[str,Any]):
            html_text = self._render(ui_spec, rows)
            return (html_text, claims)

        return self.responder.respond(ctx=ctx, generate=_gen)