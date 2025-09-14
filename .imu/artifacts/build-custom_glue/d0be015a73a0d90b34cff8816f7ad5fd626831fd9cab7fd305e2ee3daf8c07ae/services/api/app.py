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

class User(BaseModel):
    id: int
    age: int
    height_cm: int
    weight_kg: float
    level: str

DB_USER: Dict[int, User] = {}

@app.post('/users')
def create_user(obj: User):
    if obj.id in DB_USER:
        raise HTTPException(409, 'id exists')
    DB_USER[obj.id] = obj
    return {'ok': True, 'user': obj}
@app.get('/users/{oid}')
def get_user(oid: int):
    if oid not in DB_USER:
        raise HTTPException(404, 'not found')
    return DB_USER[oid]
@app.get('/users')
def list_user():
    return list(DB_USER.values())

class WorkoutSession(BaseModel):
    id: int
    user_id: int
    date: str
    minutes: int
    intensity: int
    type: str

DB_WORKOUTSESSION: Dict[int, WorkoutSession] = {}

@app.post('/workoutsessions')
def create_workoutsession(obj: WorkoutSession):
    if obj.id in DB_WORKOUTSESSION:
        raise HTTPException(409, 'id exists')
    DB_WORKOUTSESSION[obj.id] = obj
    return {'ok': True, 'workoutsession': obj}
@app.get('/workoutsessions/{oid}')
def get_workoutsession(oid: int):
    if oid not in DB_WORKOUTSESSION:
        raise HTTPException(404, 'not found')
    return DB_WORKOUTSESSION[oid]
@app.get('/workoutsessions')
def list_workoutsession():
    return list(DB_WORKOUTSESSION.values())

class Plan(BaseModel):
    id: int
    user_id: int
    start_date: str
    weeks: int
    goal: str
    weekly_minutes: int

DB_PLAN: Dict[int, Plan] = {}

@app.post('/plans')
def create_plan(obj: Plan):
    if obj.id in DB_PLAN:
        raise HTTPException(409, 'id exists')
    DB_PLAN[obj.id] = obj
    return {'ok': True, 'plan': obj}
@app.get('/plans/{oid}')
def get_plan(oid: int):
    if oid not in DB_PLAN:
        raise HTTPException(404, 'not found')
    return DB_PLAN[oid]
@app.get('/plans')
def list_plan():
    return list(DB_PLAN.values())

class Recommendation(BaseModel):
    id: int
    user_id: int
    plan_id: int
    next_session_type: str
    target_minutes: int
    rationale: str

DB_RECOMMENDATION: Dict[int, Recommendation] = {}

@app.post('/recommendations')
def create_recommendation(obj: Recommendation):
    if obj.id in DB_RECOMMENDATION:
        raise HTTPException(409, 'id exists')
    DB_RECOMMENDATION[obj.id] = obj
    return {'ok': True, 'recommendation': obj}
@app.get('/recommendations/{oid}')
def get_recommendation(oid: int):
    if oid not in DB_RECOMMENDATION:
        raise HTTPException(404, 'not found')
    return DB_RECOMMENDATION[oid]
@app.get('/recommendations')
def list_recommendation():
    return list(DB_RECOMMENDATION.values())

class ComputeIn(BaseModel):
    productId: float = 0.0

WEIGHTS_CALCULATETOTALPRICE = [1, 1]
INPUTS_CALCULATETOTALPRICE  = ["productId"]

@app.post('/compute/CalculateTotalPrice')
def compute(inp: ComputeIn):
    xs = [getattr(inp, k, 0.0) for k in INPUTS_CALCULATETOTALPRICE]
    ws = WEIGHTS_CALCULATETOTALPRICE
    if len(xs) != len(ws):
        raise HTTPException(422, 'weights mismatch')
    score = sum(x*w for x, w in zip(xs, ws))
    return {'ok': True, 'score': score}

@app.get('/')
def root():
    return {'ok': True, 'entities': ["User", "WorkoutSession", "Plan", "Recommendation"] }
