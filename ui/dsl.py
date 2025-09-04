# imu_repo/ui/dsl.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal

ComponentKind = Literal[
    "text","input","button","list","image","spacer","container",
    "form","fieldset","select","checkbox","radio","table","markdown",
    "grid","col"  # NEW
]

@dataclass
class Component:
    kind: ComponentKind
    id: str
    props: Dict[str, Any] = field(default_factory=dict)
    children: List["Component"] = field(default_factory=list)

@dataclass
class Page:
    title: str
    components: List[Component] = field(default_factory=list)
    theme: Dict[str, Any] = field(default_factory=dict)
    permissions: Dict[str, bool] = field(default_factory=dict)

class DSLValidationError(Exception): ...

_ALLOWED_IMAGE_SCHEMES = ("data:",)

def _valid_id(s: str) -> bool:
    return bool(s) and s.replace("-","_").isalnum()

def _require(cond: bool, msg: str):
    if not cond: raise DSLValidationError(msg)

def _is_int(v) -> bool:
    try:
        int(v)
        return True
    except Exception:
        return False

def validate_page(page: Page) -> None:
    _require(isinstance(page.title, str) and page.title, "page.title required")
    seen = set()

    def walk(c: Component, parent: Optional[Component]):
        _require(_valid_id(c.id), f"invalid id: {c.id}")
        _require(c.id not in seen, f"duplicate id: {c.id}")
        seen.add(c.id)

        if c.kind == "text":
            _require("text" in c.props and isinstance(c.props["text"], str), "text requires 'text'")
        elif c.kind == "input":
            tp = c.props.get("type","text")
            _require(tp in ("text","email","number","password","search"), f"unsupported input.type: {tp}")
        elif c.kind == "button":
            _require("label" in c.props, "button requires 'label'")
            _require(isinstance(c.props.get("action",""), str), "button.action must be string")
        elif c.kind == "list":
            items = c.props.get("items",[])
            _require(isinstance(items, list) and all(isinstance(i,str) for i in items), "list.items must be list[str]")
        elif c.kind == "image":
            src = c.props.get("src","")
            _require(any(src.startswith(pre) for pre in _ALLOWED_IMAGE_SCHEMES), "image.src must be data: URI")
        elif c.kind == "container":
            pass
        elif c.kind == "spacer":
            _ = int(c.props.get("h", 12))
        elif c.kind == "markdown":
            _require("md" in c.props and isinstance(c.props["md"], str), "markdown requires 'md'")
        elif c.kind == "form":
            _require(isinstance(c.props.get("schema", {}), dict), "form requires schema dict")
            _require(isinstance(c.props.get("submit_label","Submit"), str), "form.submit_label must be str")
        elif c.kind == "fieldset":
            legend = c.props.get("legend","")
            _require(isinstance(legend, str), "fieldset.legend must be str")
            _require(parent is not None and parent.kind in ("form","fieldset","container"), "fieldset must be under form/fieldset/container")
        elif c.kind == "select":
            opts = c.props.get("options",[])
            _require(isinstance(opts, list) and all(isinstance(o,(str,dict)) for o in opts), "select.options must be list[str|dict]")
        elif c.kind == "checkbox":
            _require(isinstance(c.props.get("label",""), str), "checkbox.label must be str")
        elif c.kind == "radio":
            name = c.props.get("name","")
            _require(isinstance(name, str) and name, "radio.name required")
            _require(isinstance(c.props.get("label",""), str), "radio.label must be str")
        elif c.kind == "table":
            cols = c.props.get("columns",[])
            rows = c.props.get("rows",[])
            _require(isinstance(cols, list) and all(isinstance(x,str) for x in cols), "table.columns must be list[str]")
            _require(isinstance(rows, list) and all(isinstance(r, list) for r in rows), "table.rows must be list[list]")
            _require(isinstance(c.props.get("filter", False), (bool,int)), "table.filter must be bool")
            _require(isinstance(c.props.get("sortable", True), (bool,int)), "table.sortable must be bool")
        elif c.kind == "grid":
            # props: cols (int, default 12), gap (px int), breakpoints (dict of min-width)
            cols = c.props.get("cols", 12)
            _require(_is_int(cols) and 1 <= int(cols) <= 48, "grid.cols must be 1..48")
            gap = c.props.get("gap", 12)
            _require(_is_int(gap) and 0 <= int(gap) <= 96, "grid.gap must be 0..96")
            bps = c.props.get("breakpoints", {"sm": 480, "md": 768, "lg": 1024, "xl": 1440})
            _require(isinstance(bps, dict) and all(_is_int(v) for v in bps.values()), "grid.breakpoints must be dict[str->int]")
        elif c.kind == "col":
            # props: span (int or dict per breakpoint)
            span = c.props.get("span", 12)
            if isinstance(span, dict):
                _require(all(_is_int(v) for v in span.values()), "col.span dict values must be int")
            else:
                _require(_is_int(span), "col.span must be int or dict")
            _require(parent is not None and parent.kind == "grid", "col must be under grid")
        else:
            raise DSLValidationError(f"unsupported kind: {c.kind}")

        for ch in c.children:
            walk(ch, c)

    for comp in page.components:
        walk(comp, None)