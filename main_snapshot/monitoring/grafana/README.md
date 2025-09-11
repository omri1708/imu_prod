# Grafana Dashboards (IMU)

- `imu_api.json` – Latency p95 (ms), Error rate (%)
- `imu_ws.json` – WS connections, WFQ queue size
- `imu_scheduler.json` – Jobs success/fail rates, job p95 runtime

## Deploy via Helm (control-plane chart)
Set `dashboards.enabled: true` and Grafana will load these as ConfigMaps (assumes sidecar/dashboards).



