# imu_repo/engine/pipeline_defaults.py
from __future__ import annotations
from typing import Callable, Awaitable, Any, Optional, Dict

from engine.strict_grounded import strict_guarded_for_user

async def build_user_guarded(handler: Callable[[Any], Awaitable[Any]]|Callable[[Any], Any],
                             *,
                             user_id: Optional[str]) -> Callable[[Any], Awaitable[Dict[str,Any]]]:
    """
    נקודת כניסה אחידה ל-Pipeline: כל handler חייב לעבור strict grounded per-user.
    """
    return await strict_guarded_for_user(handler, user_id=user_id)