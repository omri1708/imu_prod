from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any

app = FastAPI(title='Domain App (generic-intelligent)')

# no core_behavior inputs provided; compute endpoint skipped