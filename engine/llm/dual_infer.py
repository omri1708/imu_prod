from __future__ import annotations
import time
from typing import Dict, Any, Callable

def run_dual(fnA: Callable[[], str], fnB: Callable[[], str]) -> Dict[str,Any]:
    t0=time.time(); a=fnA(); tA=(time.time()-t0)*1000.0
    t0=time.time(); b=fnB(); tB=(time.time()-t0)*1000.0
    return {"A":{"text":a,"lat_ms":tA}, "B":{"text":b,"lat_ms":tB}}
