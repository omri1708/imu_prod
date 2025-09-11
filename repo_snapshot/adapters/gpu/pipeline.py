# adapters/gpu/pipeline.py
# -*- coding: utf-8 -*-
from typing import List, Callable
from ..contracts import ResourceRequired
from .cuda_runner import compile_and_run_cuda

def run_jobs_multi_gpu(num_devices: int, job_builder: Callable[[], str]) -> List[str]:
    """
    job_builder מחזיר קוד/משימה; כאן לצורך דוגמה נריץ את הסמפלים שלנו.
    """
    try:
        r = compile_and_run_cuda(devices=num_devices)
    except ResourceRequired as e:
        raise
    else:
        return r["results"]
