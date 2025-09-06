# server/prom_anomaly.py
# Spike detector מינימלי לשילוב בתוך Auto-Canary (על נתוני Prometheus):
from __future__ import annotations
from typing import List, Dict, Any, Optional
import math

def ema(series: List[float], alpha: float = 0.2) -> List[float]:
    if not series: return []
    out=[series[0]]
    for x in series[1:]:
        out.append(alpha*x + (1.0-alpha)*out[-1])
    return out

def stdev(series: List[float]) -> float:
    n=len(series)
    if n<2: return 0.0
    mu=sum(series)/n
    var=sum((x-mu)*(x-mu) for x in series)/(n-1)
    return math.sqrt(max(var,0.0))

def detect_spike(series: List[float], z_thresh: float = 3.0) -> Dict[str,Any]:
    """
    מחזיר {'spike':bool, 'z':z_score, 'value':last, 'mean':mu, 'stdev':sigma}
    """
    if not series: return {"spike":False,"z":0.0,"value":0.0,"mean":0.0,"stdev":0.0}
    vals=series[-60:] if len(series)>60 else series[:]  # חלון אחרון
    mu=sum(vals)/len(vals)
    sigma=stdev(vals)
    last=vals[-1]
    z=0.0 if sigma==0.0 else (last-mu)/sigma
    return {"spike": z>=z_thresh, "z": z, "value": last, "mean": mu, "stdev": sigma}

def detect_lag_spike(lat_ms_series: List[float], z_thresh: float = 3.0) -> Dict[str,Any]:
    return detect_spike(lat_ms_series, z_thresh=z_thresh)