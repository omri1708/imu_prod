# server/synth_presets_api.py
# Presets מוכנים ל-Synthesizer (Spec מלאים) — לקבל/להשתמש/לערוך ב-UI.
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter(prefix="/synth/presets", tags=["synth-presets"])

PRESETS: Dict[str, Dict[str,Any]] = {
    "pg.restore": {
      "name": "PostgreSQL Restore",
      "kind": "db.pg.restore",
      "version": "1.0.0",
      "description": "Restore via pg_restore",
      "params": {
        "db_url":     {"type":"string","required":True},
        "in":         {"type":"string","required":True},
        "clean_opt":  {"type":"string","default":""},
        "ifexists_opt":{"type":"string","default":""},
        "jobs_opt":   {"type":"string","default":""},
        "extra_opt":  {"type":"string","default":""}
      },
      "os_templates": {
        "any": "pg_restore -d {db_url}{clean_opt}{ifexists_opt}{jobs_opt}{extra_opt} {in}"
      },
      "examples": {
        "db_url":"postgres://user:pass@localhost:5432/app?sslmode=disable",
        "in":"./dump.tar",
        "clean_opt":" --clean",
        "ifexists_opt":" --if-exists",
        "jobs_opt":" -j 2",
        "extra_opt":""
      }
    },
    "kafka.produce": {
      "name": "Kafka Console Producer",
      "kind": "queue.kafka.produce",
      "version": "1.0.0",
      "description": "Produce messages to Kafka topic",
      "params": {
        "bootstrap": {"type":"string","required":True},
        "topic":     {"type":"string","required":True},
        "props_opt": {"type":"string","default":""},
        "input_opt": {"type":"string","default":""}
      },
      "os_templates": {
        "any": "kafka-console-producer --bootstrap-server {bootstrap} --topic {topic}{props_opt}{input_opt}"
      },
      "examples": {
        "bootstrap":"localhost:9092",
        "topic":"events",
        "props_opt":" --producer-property linger.ms=5",
        "input_opt":""  
      }
    },
    "terraform.apply": {
      "name": "Terraform Apply",
      "kind": "infra.terraform.apply",
      "version": "1.0.0",
      "description": "Terraform apply",
      "params": {
        "dir":        {"type":"string","required":True},
        "varfile_opt":{"type":"string","default":""},
        "var_opt":    {"type":"string","default":""},
        "backend_opt":{"type":"string","default":""},
        "auto_approve":{"type":"boolean","default":True}
      },
      "os_templates": {
        "any": "terraform -chdir={dir} apply{varfile_opt}{var_opt}{backend_opt}{approve_opt}"
      },
      "examples": {
        "dir":"./infra",
        "varfile_opt":" -var-file=prod.tfvars",
        "var_opt":"",
        "backend_opt":"",
        "approve_opt":" -auto-approve"
      }
    },
    "git.clone": {
      "name": "Git Clone",
      "kind": "scm.git.clone",
      "version": "1.0.0",
      "description": "Clone a repository",
      "params": {
        "repo":      {"type":"string","required":True},
        "dest":      {"type":"string","required":True},
        "branch_opt":{"type":"string","default":""},
        "depth_opt": {"type":"string","default":""}
      },
      "os_templates": {
        "any": "git clone {repo} {dest}{branch_opt}{depth_opt}"
      },
      "examples": {
        "repo":"https://github.com/org/repo.git",
        "dest":"./repo",
        "branch_opt":" -b main",
        "depth_opt":" --depth 1"
      }
    }
}

@router.get("/get")
def get_preset(key: str):
    p = PRESETS.get(key)
    if not p: raise HTTPException(404, f"unknown preset: {key}")
    return {"ok": True, "spec": p}

@router.get("/keys")
def keys():
    return {"ok": True, "keys": sorted(PRESETS.keys())}