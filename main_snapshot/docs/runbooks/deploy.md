# Deploy Control Plane

## One-click
- `./scripts/one_button_platform.sh`

## Manual
```bash
helm dependency build helm/control-plane
helm upgrade --install imu helm/control-plane -n default -f helm/control-plane/values.production.yaml
helm test imu -n default