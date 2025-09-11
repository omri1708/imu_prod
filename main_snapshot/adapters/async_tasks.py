# imu_repo/adapters/async_tasks.py
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any, Dict
import uuid

class AsyncError(Exception): ...

class AsyncPool:
    """
    Simple async task pool using threads.
    - submit(callable, *args, **kwargs) -> task_id
    - result(task_id, timeout=None) -> returns value or raises error
    """

    def __init__(self, max_workers: int = 8):
        self.exec = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, Future] = {}

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> str:
        if not callable(fn):
            raise AsyncError("not_callable")
        tid = uuid.uuid4().hex
        self.tasks[tid] = self.exec.submit(fn, *args, **kwargs)
        return tid

    def result(self, tid: str, timeout: float | None = None) -> Any:
        if tid not in self.tasks:
            raise AsyncError("unknown_task_id")
        fut = self.tasks[tid]
        return fut.result(timeout=timeout)
