# scripts/smoke_kind.sh
# Build->push to local registry->kind load(if needed)->helm deploy->helm test + k6 hooks->optional teardown
set -euo pipefail

CLUSTER="${CLUSTER:-imu}"
NS="${NS:-dev}"
REL="${REL:-imu}"
REG_HOST="${REG_HOST:-localhost:5001}"  # תואם kind_setup
IMG_API="${REG_HOST}/imu-api"
IMG_WS="${REG_HOST}/imu-ws"
IMG_UI="${REG_HOST}/imu-ui"
TAG="kind-$(date +%Y%m%d%H%M)-${RANDOM}"

step(){ echo -e "\n\033[1;36m==> $*\033[0m"; }

# 1) build & push
step "Build + Push images to ${REG_HOST}"
docker build -t ${IMG_API}:${TAG} -f docker/prod/api/Dockerfile .
docker build -t ${IMG_WS}:${TAG}  -f docker/prod/ws/Dockerfile  .
docker build -t ${IMG_UI}:${TAG}  -f docker/prod/ui/Dockerfile  .
docker push ${IMG_API}:${TAG} || true
docker push ${IMG_WS}:${TAG}  || true
docker push ${IMG_UI}:${TAG}  || true

# 2) values patch for dev
PATCH="helm/control-plane/values.dev.kind.yaml"
cat > "${PATCH}" <<EOF
namespace: ${NS}
images:
  api: { repository: ${IMG_API}, tag: ${TAG}, pullPolicy: IfNotPresent }
  ws:  { repository: ${IMG_WS},  tag: ${TAG}, pullPolicy: IfNotPresent }
  ui:  { repository: ${IMG_UI},  tag: ${TAG}, pullPolicy: IfNotPresent }
replicas: { api: 1, ws: 1, ui: 1 }
synthetics:
  enabled: true
  vus: 5
  duration: "20s"
  p95_ms: 800
  error_rate: 0.02
EOF

# 3) deploy
step "Helm upgrade --install"
helm upgrade --install "${REL}" helm/control-plane -n "${NS}" --create-namespace \
  -f helm/control-plane/values.yaml -f "${PATCH}"

# 4) tests
step "helm test (Chart hooks), PostSync k6 יבצע WS publish/echo"
set +e
helm test "${REL}" -n "${NS}"
RC=$?
set -e
if [ $RC -ne 0 ]; then
  echo "helm test FAILED → rollback"; helm rollback "${REL}" 1 -n "${NS}" || true; exit 1
fi

echo "SMOKE (kind) OK. set TEARDOWN=true to uninstall."
[ "${TEARDOWN:-false}" = "true" ] && helm uninstall "${REL}" -n "${NS}" || true