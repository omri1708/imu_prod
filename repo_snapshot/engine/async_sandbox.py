# imu_repo/engine/async_sandbox.py
from __future__ import annotations
import asyncio, time
from typing import Dict, Tuple
from engine.config import load_config, save_config
from engine.policy_ctx import get_user
from sandbox.limits import RateLimiter

class _Sem:
    def __init__(self, n: int):
        self.sem = asyncio.Semaphore(max(1, int(n)))

    async def __aenter__(self):
        await self.sem.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.sem.release()

class AsyncCaps:
    def __init__(self):
        self._g_sem: _Sem | None = None
        self._u_sem: Dict[str, _Sem] = {}
        self._c_sem: Dict[Tuple[str,str], _Sem] = {}
        self._rl_caps: Dict[str, RateLimiter] = {}  # per-capability limiter

    def _cfg(self):
        cfg = load_config()
        a = dict(cfg.get("async", {}))
        a.setdefault("max_global", 32)
        a.setdefault("per_user", 8)
        a.setdefault("per_capability", {})  # {"text.gen": 4, "vec.search": 16}
        a.setdefault("per_capability_rps", {})  # {"text.gen": {"rps":2.5,"burst":3}}
        cfg["async"] = a
        save_config(cfg)
        return a

    def _global_sem(self) -> _Sem:
        if self._g_sem is None:
            a = self._cfg()
            self._g_sem = _Sem(a["max_global"])
        return self._g_sem

    def _user_sem(self, user_id: str) -> _Sem:
        a = self._cfg()
        s = self._u_sem.get(user_id)
        if s is None:
            s = _Sem(a["per_user"])
            self._u_sem[user_id] = s
        return s

    def _cap_sem(self, user_id: str, capability: str) -> _Sem:
        a = self._cfg()
        key = (user_id, capability)
        s = self._c_sem.get(key)
        if s is None:
            limit = int(a["per_capability"].get(capability, a["per_user"]))
            s = _Sem(limit)
            self._c_sem[key] = s
        return s

    def _cap_rl(self, capability: str) -> RateLimiter:
        a = self._cfg()
        r = self._rl_caps.get(capability)
        if r is None:
            spec = a["per_capability_rps"].get(capability, {"rps": 10.0, "burst": 5})
            r = RateLimiter(rate_per_sec=float(spec.get("rps", 10.0)), burst=int(spec.get("burst", 5)))
            self._rl_caps[capability] = r
        return r

    async def enter(self, capability: str):
        uid = get_user() or "anon"
        g = self._global_sem()
        u = self._user_sem(uid)
        c = self._cap_sem(uid, capability)
        # סדר עקבי כדי למנוע deadlock: global → user → user+cap
        await g.__aenter__()
        await u.__aenter__()
        await c.__aenter__()
        # Rate-limit per capability per host==cap (לוגית)
        await self._cap_rl(capability).acquire(uid, capability, amount=1.0)
        return g, u, c

# singleton
ASYNC_CAPS = AsyncCaps()