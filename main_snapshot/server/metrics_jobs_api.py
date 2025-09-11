# server/metrics_jobs_api.py
# API: /metrics/jobs/summary?hours=24
from __future__ import annotations
from fastapi import APIRouter, Query
from typing import Dict, Any
from .job_runs import summary

router = APIRouter(prefix="/metrics/jobs", tags=["metrics"])

@router.get("/summary")
def jobs_summary(hours: int = Query(24, ge=1, le=720)) -> Dict[str,Any]:
    return summary(hours=hours)