# imu_repo/orchestration/compose_workflow.py
from __future__ import annotations
import os, time
from typing import Dict, Any
from orchestration.docker_compose import write_compose, up, down, has_docker, ResourceRequired
from orchestration.services import ServiceSpec
from orchestration.docker_compose import ComposeWriter, ResourceRequired

COMPOSE_PATH=".imu_state/compose/redis.yml"


def ensure_redis_compose(up_mode: bool = True) -> bool:
    """
    ייצור והרמה של redis:7 אם Docker זמין.
    מחזיר True אם Redis אמור לרוץ בלוקאל.
    """
    if not has_docker(): return False
    services={
        "redis": {
            "image":"redis:7",
            "ports":["6379:6379"]
        }
    }
    write_compose(COMPOSE_PATH, services)
    if up_mode:
        up(COMPOSE_PATH)
        time.sleep(1.0)
        return True
    return False


def shutdown_redis_compose():
    if has_docker():
        try: down(COMPOSE_PATH)
        except Exception: pass


def build_stack() -> str:
    # נבנה web (nginx), redis, ו-worker (busybox המדמה קרון)
    web = ServiceSpec(
        name="web",
        image="nginx:alpine",
        ports=["8080:80"],
    )
    redis = ServiceSpec(
        name="redis",
        image="redis:7-alpine",
        ports=["6379:6379"]
    )
    worker = ServiceSpec(
        name="worker",
        image="busybox:stable",
        command=["sh","-c","while true; do echo tick; sleep 5; done"],
        depends_on=["redis"]
    )
    cw=ComposeWriter()
    path=cw.write([web, redis, worker])
    try:
        cw.up(path)
        return "UP:"+path
    except ResourceRequired as rr:
        return "NEED:"+rr.how


def teardown(compose_path: str):
    cw=ComposeWriter()
    cw.down(compose_path)