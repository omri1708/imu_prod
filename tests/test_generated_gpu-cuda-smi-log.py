# tests/test_generated_gpu-cuda-smi-log.py
from fastapi.testclient import TestClient
from server.http_api import APP
client=TestClient(APP)

def test_cuda_smi_log_dryrun():
    params={"interval_s":2,"query":"utilization.gpu,utilization.memory","format_opt":" --format=csv,noheader,nounits","count_opt":""}
    r=client.post("/adapters/dry_run", json={"user_id":"demo-user","kind":"gpu.cuda.smi-log","params":params})
    assert r.status_code==200
    j=r.json(); assert j["ok"] and "nvidia-smi --query-gpu=utilization.gpu,utilization.memory" in j["cmd"]