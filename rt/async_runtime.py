# imu_repo/rt/async_runtime.py
from __future__ import annotations
import asyncio, contextlib, time
from typing import Callable, Awaitable, Any, Dict

class TaskFailed(Exception): ...
class DeadlineExceeded(TaskFailed): ...
class Cancelled(TaskFailed): ...

class AsyncSupervisor:
    """
    מריץ קורוטינות עם deadline, retry ו- jitter מבוקר.
    - run: הרצה יחידה עם deadline.
    - retry: הרצה עם ניסיונות חוזרים אקספוננציאליים.
    """
    def __init__(self, *, default_deadline_s: float = 10.0, max_concurrency: int = 100):
        self.default_deadline_s = float(default_deadline_s)
        self._sem = asyncio.Semaphore(max_concurrency)

    async def run(self, coro: Awaitable[Any], *, deadline_s: float | None = None) -> Any:
        deadline_s = self.default_deadline_s if deadline_s is None else float(deadline_s)
        async with self._sem:
            try:
                return await asyncio.wait_for(coro, timeout=deadline_s)
            except asyncio.TimeoutError as e:
                raise DeadlineExceeded(str(e))
            except asyncio.CancelledError as e:
                raise Cancelled(str(e))

    async def retry(self, factory: Callable[[], Awaitable[Any]], *,
                    attempts: int = 5, initial_backoff_s: float = 0.05,
                    deadline_s: float | None = None) -> Any:
        last_exc = None
        for i in range(attempts):
            try:
                return await self.run(factory(), deadline_s=deadline_s)
            except (DeadlineExceeded, Cancelled, Exception) as e:
                last_exc = e
                await asyncio.sleep(min(1.0, initial_backoff_s * (2 ** i)))
        raise TaskFailed(f"retry_exhausted: {last_exc}")