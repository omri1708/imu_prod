# server/prometheus_client.py
# לקוח HTTP מינימלי ל-Prometheus (instant/range) – ללא תלות חיצונית.
from __future__ import annotations
from typing import Dict, Any
import urllib.parse, urllib.request, json, time

def query_instant(base_url: str, promql: str, ts: float | None = None, timeout: int = 8) -> Dict[str,Any]:
    ts = ts or time.time()
    url = f"{base_url.rstrip('/')}/api/v1/query?{urllib.parse.urlencode({'query': promql, 'time': ts})}"
    req = urllib.request.Request(url, headers={"User-Agent":"imu-prom"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def query_range(base_url: str, promql: str, start: float, end: float, step: float = 30.0, timeout: int = 8) -> Dict[str,Any]:
    url = f"{base_url.rstrip('/')}/api/v1/query_range?{urllib.parse.urlencode({'query': promql, 'start': start, 'end': end, 'step': step})}"
    req = urllib.request.Request(url, headers={"User-Agent":"imu-prom"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def extract_last_vector(resp: Dict[str,Any]) -> float | None:
    # instant vector
    try:
        res = resp["data"]["result"]
        if not res: return None
        v = res[0]["value"][1]
        return float(v)
    except Exception:
        return None

def quantile_from_range(resp: Dict[str,Any], q: float) -> float | None:
    # חישוב קוונטיל פשוט על טווח (דוגם מהvalues)
    try:
        res = resp["data"]["result"]
        if not res: return None
        series = res[0]["values"]
        vals = sorted(float(v[1]) for v in series if v and v[1] is not None)
        if not vals: return None
        idx = int(q * (len(vals)-1))
        return float(vals[idx])
    except Exception:
        return None