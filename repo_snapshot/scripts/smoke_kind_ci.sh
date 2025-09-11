# scripts/smoke_kind_ci.sh
# CI Smoke on Kind: build images -> kind load -> helm deploy -> helm test + k6 hooks (rollback on fail)
set -euo pipefail

CLUSTER="${CLUSTER:-imu}"
NS="${NS:-dev}"
REL="${REL:-imu}"
IMG_API="imu-api"
IMG_WS="imu-ws"
IMG_UI="imu-ui"
TAG="ci-$(date +%Y%m%d%H%M)-${RANDOM}"

step(){ echo -e "\n\033[1;36m==> $*\033[0m"; }

build(){
  step "Build images (tag=${TAG})"
  docker build -t ${IMG_API}:${TAG} -f docker/prod/api/Dockerfile .
  docker build -t ${IMG_WS}:${TAG}  -f docker/prod/ws/Dockerfile  .
  docker build -t ${IMG_UI}:${TAG}  -f docker/prod/ui/Dockerfile  .
}

kind_load(){
  step "kind load images"
  kind load docker-image ${IMG_API}:${TAG} --name "${CLUSTER}"
  kind load docker-image ${IMG_WS}:${TAG}  --name "${CLUSTER}"
  kind load docker-image ${IMG_UI}:${TAG}  --name "${CLUSTER}"
}

patch_values(){
  local f="helm/control-plane/values.dev.kind-ci.yaml"
  step "Patch values -> $f"
  cat > "$f" <<EOF
namespace: ${NS}
images:
  api: { repository: ${IMG_API}, tag: ${TAG}, pullPolicy: IfNotPresent }
  ws:  { repository: ${IMG_WS},  tag: ${TAG}, pullPolicy: IfNotPresent }
  ui:  { repository: ${IMG_UI},  tag: ${TAG}, pullPolicy: IfNotPresent }
replicas: { api: 1, ws: 1, ui: 1 }
synthetics: { enabled: true, vus: 5, duration: "15s", p95_ms: 900, error_rate: 0.05 }
observability: { pushgateway: { enabled: false, url: "" } }
EOF
  echo "$f"
}

deploy(){
  local patched="$1"
  step "helm upgrade --install control-plane"
  helm upgrade --install "${REL}" helm/control-plane -n "${NS}" --create-namespace \
    -f helm/control-plane/values.yaml -f "${patched}"
}

tests(){
  step "helm test ${REL} -n ${NS}"
  set +e
  helm test "${REL}" -n "${NS}"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "Helm test FAILED — rollback"
    helm rollback "${REL}" 1 -n "${NS}" || true
    kubectl -n "${NS}" get pods
    kubectl -n "${NS}" logs jobs/${REL}-postsync-k6 -p --tail=-1 || true
    exit 1
  fi
  echo "OK"
}

main(){
  build
  kind_load
  patched=$(patch_values)
  deploy "$patched"
  tests
}
main "$@"