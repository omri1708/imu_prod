# scripts/smoke_all.sh
# Smoke E2E: build dev images -> patch values.dev -> deploy -> helm test (+k6 hooks) -> (optional teardown)
set -euo pipefail

NS="${NS:-dev}"
REL="${REL:-imu}"
VALUES="${VALUES:-helm/control-plane/values.yaml}"            # בסיס (אפשר להחליף ל-values.dev.yaml)
CP_CHART="helm/control-plane"
IMG_OWNER="${IMG_OWNER:-local}"                               # docker.io/<owner>/...
IMG_TAG="dev-$(date +%Y%m%d%H%M)-${RANDOM}"

step(){ echo -e "\n\033[1;36m==> $*\033[0m"; }

build_images(){
  step "Build dev images (tag=$IMG_TAG)"
  docker build -t ${IMG_OWNER}/imu-api:${IMG_TAG} -f docker/prod/api/Dockerfile .
  docker build -t ${IMG_OWNER}/imu-ws:${IMG_TAG}  -f docker/prod/ws/Dockerfile  .
  docker build -t ${IMG_OWNER}/imu-ui:${IMG_TAG}  -f docker/prod/ui/Dockerfile  .
  echo "Images built locally with tag ${IMG_TAG}"
}

patch_values_dev(){
  local file="helm/control-plane/values.dev.patched.yaml"
  step "Patch values for dev images -> $file"
  cat > "$file" <<EOF
namespace: ${NS}
images:
  api: { repository: ${IMG_OWNER}/imu-api, tag: ${IMG_TAG}, pullPolicy: IfNotPresent }
  ws:  { repository: ${IMG_OWNER}/imu-ws,  tag: ${IMG_TAG}, pullPolicy: IfNotPresent }
  ui:  { repository: ${IMG_OWNER}/imu-ui,  tag: ${IMG_TAG}, pullPolicy: IfNotPresent }
replicas: { api: 1, ws: 1, ui: 1 }
synthetics: { enabled: true, vus: 5, duration: "20s", p95_ms: 800, error_rate: 0.02 }
EOF
  echo "$file"
}

deploy(){
  local patched="$1"
  step "helm upgrade --install ${REL} ${CP_CHART} -n ${NS}"
  helm upgrade --install "${REL}" "${CP_CHART}" -n "${NS}" -f "${VALUES}" -f "${patched}" --create-namespace
}

tests(){
  step "helm test ${REL} -n ${NS} (Chart tests)"
  set +e
  helm test "${REL}" -n "${NS}"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "Helm tests FAILED — rollback"
    helm rollback "${REL}" 1 -n "${NS}" || true
    exit 1
  fi
  echo "Helm tests PASSED"
}

teardown(){
  step "Teardown? (TEARDOWN=true to enable)"
  if [[ "${TEARDOWN:-false}" == "true" ]]; then
    helm uninstall "${REL}" -n "${NS}" || true
  fi
}

main(){
  build_images
  patched=$(patch_values_dev)
  deploy "$patched"
  tests
  step "Smoke complete (namespace=${NS}, release=${REL})"
  teardown
}
main "$@"