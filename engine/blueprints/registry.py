# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, Dict, Any
from engine.blueprints.generic_backend import  generate_backend as generic_backend

REGISTRY = {}

def resolve(domain: str) -> Callable[[Dict[str,Any]], Dict[str,bytes]]:
    return REGISTRY.get(domain, generic_backend)