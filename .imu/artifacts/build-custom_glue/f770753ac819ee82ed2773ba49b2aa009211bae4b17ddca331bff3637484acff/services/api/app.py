import os, logging, time
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

LOG_LEVEL = os.getenv('LOG_LEVEL','INFO').upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
app = FastAPI(title='IMU Domain Backend')

@app.get('/healthz')
def healthz():
    return {'ok': True, 'ts': time.time()}

@app.get('/readyz')
def readyz():
    return {'ready': True}

@app.get('/metrics')
def metrics():
    # minimal Prometheus plaintext without external deps
    body = '# HELP app_up 1 if app is up\n# TYPE app_up gauge\napp_up 1\n'
    return (body, 200, {'Content-Type': 'text/plain; version=0.0.4'})

class ComputeIn(BaseModel):
    orderId: float = 0.0

WEIGHTS_CALCULATETOTALPRICE = [1.0]
INPUTS_CALCULATETOTALPRICE  = ["orderId"]

@app.post('/compute/calculateTotalPrice')
def compute(inp: ComputeIn):
    xs = [getattr(inp, k, 0.0) for k in INPUTS_CALCULATETOTALPRICE]
    ws = WEIGHTS_CALCULATETOTALPRICE
    if len(xs) != len(ws):
        raise HTTPException(422, 'weights mismatch')
    score = sum(x*w for x, w in zip(xs, ws))
    return {'ok': True, 'score': score}

@app.get('/')
def root():
    return {'ok': True, 'entities': ["Entity"] }
