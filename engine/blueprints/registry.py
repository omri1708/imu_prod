# PATH: engine/blueprints/registry.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from .generic_backend import generate_backend
from .iac_terraform import generate as generate_iac_terraform
from .ci_github_actions import generate as generate_ci_github_actions
from .market_scan import generate as generate_market_scan 
from typing import Callable, Dict, Any

"""
Legal / Use Anchor:
- Lawful blueprint registry; no ToS violations.
"""



_REGISTRY = {
    "generic_backend": generate_backend,
    "iac.terraform": generate_iac_terraform,
    "ci.github_actions": generate_ci_github_actions,
    "market.scan": generate_market_scan,
}


def resolve(name: str) -> Callable[[Dict[str,Any]], Dict[str,bytes]]:
    return _REGISTRY.get(name, generate_backend)
