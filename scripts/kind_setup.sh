# scripts/kind_setup.sh
# יוצרת kind cluster עם registry מקומי (127.0.0.1:5001) ו-ingress class "nginx"
set -euo pipefail

CLUSTER="${CLUSTER:-imu}"
REG_NAME="${REG_NAME:-imu-reg}"
REG_PORT="${REG_PORT:-5001}"

if ! command -v kind >/dev/null 2>&1; then
  echo "kind not found"; exit 1
fi

running="$(docker ps -q -f name=${REG_NAME})"
if [ -z "$running" ]; then
  echo ">> starting local registry at 127.0.0.1:${REG_PORT}"
  docker run -d --restart=always -p "${REG_PORT}:5000" --name "${REG_NAME}" registry:2
fi

cat <<EOF | kind create cluster --name "${CLUSTER}" --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
containerdConfigPatches:
- |-
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors."localhost:${REG_PORT}"]
    endpoint = ["http://${REG_NAME}:5000"]
nodes:
- role: control-plane
- role: worker
- role: worker
EOF

echo ">> connecting registry to network"
docker network connect "kind" "${REG_NAME}" || true

echo ">> documenting local registry"
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: local-registry-hosting
  namespace: kube-public
data:
  localRegistryHosting.v1: |
    host: "localhost:${REG_PORT}"
    help: "https://kind.sigs.k8s.io/docs/user/local-registry/"
EOF

echo "OK: kind=${CLUSTER}, registry=localhost:${REG_PORT}"