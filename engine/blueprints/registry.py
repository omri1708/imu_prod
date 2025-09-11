# PATH: engine/blueprints/registry.py
from __future__ import annotations
from typing import Callable, Dict, Any, Type
import importlib, pkgutil, inspect
from importlib import import_module

# --- פונקציות "ידועות" בשם קבוע (רישום מוקדם) ---
from engine.blueprints.generic_backend import generate_backend as generate_api
from engine.blueprints.iac_terraform import generate as generate_iac_terraform
from engine.blueprints.ci_github_actions import generate as generate_ci_github_actions
from engine.blueprints.market_scan import generate as generate_market_scan
from engine.blueprints.auto_writer import synthesize_blueprint

_REGISTRY: Dict[str, Callable[[Dict[str,Any]], Dict[str,bytes]]] = {
    "generic_backend": generate_api,
    "api": generate_api,
    "custom": generate_api,
    "iac.terraform": generate_iac_terraform,
    "ci.github_actions": generate_ci_github_actions,
    "market.scan": generate_market_scan,
}

# --- טעינת מחלקות Blueprint (אם יש כאלה עם NAME ייחודי) ---
def load_registry() -> Dict[str, type]:
    import engine.blueprints as pkg
    reg: Dict[str, type] = {}
    for modinfo in pkgutil.iter_modules(pkg.__path__):
        name = modinfo.name
        if name.startswith("_"):
            continue
        mod = importlib.import_module(f"engine.blueprints.{name}")
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if getattr(cls, "NAME", None):
                reg[cls.NAME] = cls
    return reg

REGISTRY = load_registry()  # אופציונלי – לשימוש קיים במקומות אחרים

def get_blueprint(name: str):  # תאימות לאחור
    return REGISTRY[name]

# --- פתרון דינמי לשמות נקודתיים + Auto-Writer ---
def _dynamic_load(name: str) -> Callable[[Dict[str,Any]], Dict[str,bytes]]:
    mod_name = f"engine.blueprints.{name.replace('.','_')}"
    try:
        mod = import_module(mod_name)
    except Exception:
        # אם לא קיים—נייצר קובץ באמצעות האוטו-רייטר ואז נטען
        synthesize_blueprint(name, {})
        mod = import_module(mod_name)
    fn = getattr(mod, "generate")
    _REGISTRY[name] = fn
    return fn

def resolve(name: str) -> Callable[[Dict[str,Any]], Dict[str,bytes]]:
    return _REGISTRY.get(name) or _dynamic_load(name)
