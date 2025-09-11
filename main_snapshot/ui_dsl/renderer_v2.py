# imu_repo/ui_dsl/renderer_v2.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable, Optional
import json
from engine.respond_strict import RespondStrict
from ui_dsl.provenance import build_ui_provenance
from ui_dsl.advanced_components import render_grid, render_table_advanced
from ui_dsl.versioning import app_version_manifest
from cas.store import put_bytes

DataProvider = Callable[[Dict[str,Any]], Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]]

class AdvancedStrictUIRenderer:
    """
    מרנדר UI-DSL מתקדם:
      - תומך grid (areas/nested) + טבלה מתקדמת (filters/sort/freeze).
      - יוצר Provenance למבנה ה-UI ול-assets.
      - יוצר גרסת אפליקציה (version manifest) ונשען עליה כ-claim נוסף.
      - אורז פלט תחת RespondStrict (Grounded-Strict Everywhere).
    """
    def __init__(self, *, base_policy: Dict[str,Any], http_fetcher=None, sign_key_id: str="root"):
        self.responder = RespondStrict(base_policy=base_policy, http_fetcher=http_fetcher, sign_key_id=sign_key_id)

    def _render_component(self, spec: Dict[str,Any], rows: List[Dict[str,Any]]) -> str:
        t = (spec or {}).get("type","table")
        if t == "grid":
            children = spec.get("children") or {}
            rendered_children = {}
            # תמיכה ב-nested: אם ילד הוא מפרט, מרנדרים כראוי; אם מחרוזת HTML – משאירים
            for name, child in children.items():
                if isinstance(child, dict) and child.get("type"):
                    if child["type"] == "table":
                        rendered_children[name] = render_table_advanced(child, rows)
                    elif child["type"] == "grid":
                        rendered_children[name] = self._render_component(child, rows)
                    else:
                        rendered_children[name] = f"<div>unsupported child component: {child['type']}</div>"
                elif isinstance(child, str):
                    rendered_children[name] = child
                else:
                    rendered_children[name] = "<div/>"
            return render_grid(spec, rendered_children)
        elif t == "table":
            return render_table_advanced(spec, rows)
        else:
            return "<div>unsupported ui component</div>"

    def render_and_package(self,
                           *,
                           ctx: Dict[str,Any],
                           ui_spec: Dict[str,Any],
                           data_provider: DataProvider) -> Dict[str,Any]:
        rows, data_claims = data_provider(ctx)

        # 1) רישום assets (אם קיימים ב-ui_spec) ושיוכם ל-Provenance
        #    build_ui_provenance כבר עושה put_bytes ל-assets שהוכנסו כ-bytes במפרט.
        sources = []
        for c in data_claims:
            for ev in c.get("evidence", []):
                if isinstance(ev, dict): sources.append(ev)
        prov = build_ui_provenance(ui_spec=ui_spec, sources=sources, policy=self.responder.base_policy)

        # 2) גרסת אפליקציה (version manifest) — עוזר לדה-דופ/קאשינג/עקיבות
        assets_meta = prov["manifest"].get("assets") or []
        version = app_version_manifest(ui_spec=ui_spec, assets=assets_meta, policy=self.responder.base_policy)

        # 3) מרנדרים HTML מלא
        html_text = self._render_component(ui_spec, rows)

        # 4) claims:
        claims = list(data_claims)
        claims.append({
            "id": f"ui:{prov['manifest_sha256'][:16]}",
            "type": "ui_provenance",
            "text": "ui spec & assets integrity",
            "schema": {"type":"manifest","unit":""},
            "value": prov["manifest_sha256"],
            "evidence": [{"kind":"cas_manifest","sha256": prov["manifest_sha256"]}],
            "consistency_group": "ui"
        })
        claims.append({
            "id": f"ui_ver:{version['sha256'][:16]}",
            "type": "ui_version",
            "text": "ui app version",
            "schema": {"type":"hash","unit":"sha256"},
            "value": version["sha256"],
            "evidence": [{"kind":"cas_manifest","sha256": version["manifest_sha256"]}],
            "consistency_group": "ui"
        })

        # 5) אריזה תחת Grounded-Strict
        def _gen(_ctx: Dict[str,Any]):
            return (html_text, claims)

        return self.responder.respond(ctx=ctx, generate=_gen)