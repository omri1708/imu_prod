# tests/test_generated_gpu-nvml-metrics.py
from fastapi.testclient import TestClient
from server.http_api import APP
client = TestClient(APP)

def test_nvml_metrics_dryrun():
    params={"query":"utilization.gpu,memory.used","format_opt":" --format=csv,noheader,nounits"}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"gpu.nvml.metrics","params":params})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and "nvidia-smi --query-gpu=utilization.gpu,memory.used" in j["cmd"]