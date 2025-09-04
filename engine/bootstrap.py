# imu_repo/engine/bootstrap.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any

from engine.pipeline import Engine
from engine.closed_loop import bootstrap_complete_system
from tests.benchmarks import default_suite

def run_bootstrap(iterations:int=3) -> Dict[str,Any]:
    eng=Engine()
    suite = default_suite()  # מגדיר תרחישי ריצה דטרמיניסטיים
    out = bootstrap_complete_system(eng, suite, iterations=iterations)
    return out

if __name__=="__main__":
    import json
    res=run_bootstrap(iterations=3)
    print(json.dumps(res, ensure_ascii=False, indent=2))
