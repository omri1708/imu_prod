# tests/test_docker_optional.py
# -*- coding: utf-8 -*-
import pytest
from adapters.contracts import ResourceRequired
from adapters.docker.build import docker_build

def test_docker_build_optional(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    (d/"Dockerfile").write_text("FROM alpine:3.19\nCMD [\"echo\",\"ok\"]\n")
    try:
        r = docker_build("imu-test:alpine", str(d))
    except ResourceRequired:
        pytest.skip("Docker not installed")
    else:
        assert r["ok"]