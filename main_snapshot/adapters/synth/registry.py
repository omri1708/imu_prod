# adapters/synth/registry.py
# Dynamic registry for generated adapters. Merges built-in templates with generated ones.
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Built-in templates (from adapters/mappings.py)
try:
    from adapters.mappings import CLI_TEMPLATES as BUILTIN_TEMPLATES
except Exception:
    BUILTIN_TEMPLATES = {}

GEN_ROOT = Path("adapters/generated")

def _scan_generated() -> Dict[str, Dict[str, str]]:
    """
    Returns {kind: {"linux": "...", "mac":"...", "win":"...", "any":"..."}}
    by scanning adapters/generated/**/cli_templates.json
    """
    res: Dict[str, Dict[str, str]] = {}
    if not GEN_ROOT.exists():
        return res
    for d in GEN_ROOT.glob("**/cli_templates.json"):
        try:
            obj = json.loads(d.read_text(encoding="utf-8"))
            # obj = {"kind":"my.kind","templates":{"linux":"...","any":"..."}}
            kind = obj.get("kind")
            templates = obj.get("templates") or {}
            if not kind or not isinstance(templates, dict): 
                continue
            res[kind] = templates
        except Exception:
            continue
    return res

_TEMPLATES: Dict[str, Dict[str, str]] = {}
_CONTRACTS: Dict[str, Path] = {}  # kind -> contract path

def _scan_contracts():
    global _CONTRACTS
    _CONTRACTS = {}
    if not GEN_ROOT.exists(): 
        return
    for d in GEN_ROOT.glob("**/contract.json"):
        try:
            # path structure: adapters/generated/<slug>/contract.json
            with open(d,"r",encoding="utf-8") as f:
                js = json.load(f)
            # find kind: stored in sibling cli_templates.json
            ct = d.parent / "cli_templates.json"
            if ct.exists():
                obj = json.loads(ct.read_text(encoding="utf-8"))
                kind = obj.get("kind")
                if kind:
                    _CONTRACTS[kind] = d
        except Exception:
            continue

def reload_registry() -> Dict[str, Dict[str, str]]:
    """Reloads generated templates and merges with built-ins"""
    global _TEMPLATES
    gen = _scan_generated()
    # merge: generated overrides built-in
    merged = dict(BUILTIN_TEMPLATES)
    for k, v in gen.items():
        merged[k] = v
    _TEMPLATES = merged
    _scan_contracts()
    return _TEMPLATES

def get_template(kind: str, fam: str) -> Optional[str]:
    """
    fam: "linux"|"mac"|"win" — falls back to "any" if present.
    """
    if not _TEMPLATES:
        reload_registry()
    t = _TEMPLATES.get(kind)
    if not t:
        return None
    if fam in t and t[fam]:
        return t[fam]
    return t.get("any")

def list_kinds() -> List[str]:
    if not _TEMPLATES:
        reload_registry()
    return sorted(_TEMPLATES.keys())

def find_contract(kind: str) -> Optional[Path]:
    if not _CONTRACTS:
        _scan_contracts()
    return _CONTRACTS.get(kind)