# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any

def generate(spec: Dict[str, Any]) -> Dict[str, bytes]:
    """
    Minimal training scaffold (placeholder). If ML training is required, replace with real pipeline.
    """
    py = b"""# training entrypoint\nif __name__=='__main__': print('training pipeline placeholder')\n"""
    return {"ml/train.py": py}
