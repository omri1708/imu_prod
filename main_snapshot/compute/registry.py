# imu_repo/compute/registry.py
from __future__ import annotations
from typing import Dict, Any, Callable, Optional, Tuple
import os, json, time

AUTOTUNE_PATH = "/mnt/data/imu_repo/autotune.json"

class Backend:
    """ממשק גנרי ל־Backend חישובי."""
    name: str = "backend"

    def supports(self, op: str, **shape: Any) -> bool:
        raise NotImplementedError

    def run(self, op: str, **kwargs: Any) -> Any:
        raise NotImplementedError

class Registry:
    def __init__(self):
        self.backends: list[Backend] = []
        self.timing: Dict[str, Dict[str, float]] = self._load_autotune()

    def _load_autotune(self) -> Dict[str, Dict[str,float]]:
        if os.path.exists(AUTOTUNE_PATH):
            try:
                return json.load(open(AUTOTUNE_PATH, "r", encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_autotune(self) -> None:
        os.makedirs(os.path.dirname(AUTOTUNE_PATH), exist_ok=True)
        tmp = AUTOTUNE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.timing, f, ensure_ascii=False, indent=2)
        os.replace(tmp, AUTOTUNE_PATH)

    def register(self, be: Backend) -> None:
        self.backends.append(be)

    def _key(self, op: str, shape: Dict[str,Any]) -> str:
        # מפתח צורה דטרמיניסטי
        items = sorted((k, str(v)) for k,v in shape.items())
        return f"{op}|" + "|".join([f"{k}={v}" for k,v in items])

    def choose_backend(self, op: str, **shape: Any) -> Optional[Backend]:
        # בחר backend מהיר עבור op+shape (אם יש טיונינג, אחר־תאימות)
        key = self._key(op, shape)
        if key in self.timing:
            # בחר את המינימום
            best = min(self.timing[key].items(), key=lambda kv: kv[1])[0]
            for b in self.backends:
                if b.name == best and b.supports(op, **shape):
                    return b
        # fallback: הראשון שתומך
        for b in self.backends:
            if b.supports(op, **shape):
                return b
        return None

    def run(self, op: str, **kwargs: Any) -> Any:
        shape = kwargs.get("_shape", {})
        be = self.choose_backend(op, **shape)
        if be is None:
            raise RuntimeError(f"no_backend_for:{op}|shape={shape}")
        t0 = time.perf_counter()
        out = be.run(op, **kwargs)
        dt = (time.perf_counter()-t0)*1000.0
        key = self._key(op, shape)
        self.timing.setdefault(key, {})
        self.timing[key][be.name] = min(dt, self.timing[key].get(be.name, dt))
        # עדכון קבוע (התכנסות למדידה המינימלית שראינו)
        self._save_autotune()
        return out

REGISTRY = Registry()