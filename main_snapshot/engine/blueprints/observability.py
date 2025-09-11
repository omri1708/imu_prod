from __future__ import annotations
from typing import Dict, Any
import json
def generate(spec: Dict[str, Any]) -> Dict[str, bytes]:
    runbook = """# Runbook
- Symptoms: High error rate / build fail / readiness not ready.
- Checks:
  - GET /healthz (expect ok=true)
  - GET /readyz (db=ok)
  - GET /metrics (Prometheus)
- Actions:
  - Restart pod / docker compose restart
  - Check CI logs
  - Rollback last deploy
"""
    rules = """
groups:
- name: imu-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 10m
    labels: { severity: page }
    annotations:
      summary: "High 5xx rate"
      description: "Rate of 5xx > 0.1 for 10m"
"""
    return {
      "observability/runbook.md": runbook.encode("utf-8"),
      "observability/prometheus-rules.yml": rules.encode("utf-8"),
      "observability/dashboard.json": json.dumps({"title":"IMU","panels":[]}, indent=2).encode("utf-8"),
    }
