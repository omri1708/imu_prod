# imu_repo/engine/keychain_manager.py
from __future__ import annotations
import time
from typing import Dict, Any, List, Callable, Tuple, Optional
from engine.key_delegation import expand_keyring_with_chain, DelegationError

class KeychainManager:
    """
    מחזיק keyring שורש + ספק שרשרת (פונקציה שמחזירה רשימת האצלות)
    ומרענן אוטומטית לפני פקיעת ה-TTL.
    """
    def __init__(self, root_keyring: Dict[str,Dict[str,str]], chain_provider: Callable[[], List[Dict[str,Any]]], *, refresh_margin_sec: int = 300):
        self._root = dict(root_keyring)
        self._prov = chain_provider
        self._ref_margin = int(refresh_margin_sec)
        self._cache: Optional[Tuple[float, List[Dict[str,Any]], Dict[str,Dict[str,str]], float]] = None
        # cache: (ts, chain, expanded, min_exp)

    def _min_exp(self, chain: List[Dict[str,Any]]) -> float:
        exps = [float(d.get("exp", 0)) for d in chain if d.get("exp")]
        return min(exps) if exps else float("inf")

    def current(self) -> Tuple[List[Dict[str,Any]], Dict[str,Dict[str,str]]]:
        now = time.time()
        if self._cache:
            ts, chain, expanded, min_exp = self._cache
            if now < (min_exp - self._ref_margin):
                return chain, expanded
        # ריענון
        chain = self._prov() or []
        expanded = expand_keyring_with_chain(self._root, chain)
        min_exp = self._min_exp(chain)
        self._cache = (now, chain, expanded, min_exp)
        return chain, expanded