# engine/adapter_types.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class AdapterResult:
    artifacts: Dict[str, str]   # path -> CAS hash
    claims: List[dict]
    evidence: List[dict]