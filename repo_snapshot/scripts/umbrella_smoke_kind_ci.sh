#!/usr/bin/env bash
# Umbrella smoke על Kind: build control-plane images -> kind load -> helm upgrade umbrella(dev)
# -> helm tests + k6 hooks של control-plane -> rollback אוטומטי בעת כשל.
set -euo pipefail

CLUSTER="${CLUSTER:-imu}"
NS="${NS:-dev}"
REL="${REL:-umbrella}"
IMG_API="imu-api"
IMG_WS="imu-ws"
IMG_UI="imu-ui"
TAG="u-ci-$(date +%Y%m%d%H%M)-${RANDOM}"

step(){ echo -e "\n\033[1;36m==> $*\033[0m"; }

build_images(){
  step "Build control-plane images (tag=${TAG})"
  docker build -t ${IMG_API}:${TAG} -f docker/prod/api/Dockerfile .
  docker build -t ${IMG_WS}:${TAG}  -f docker/prod/ws/Dockerfile  .
  docker build -t ${IMG_UI}:${TAG}  -f docker/prod/ui/Dockerfile  .
}

kind_load(){
  step "Load images to Kind cluster=${CLUSTER}"
  kind load docker-image ${IMG_API}:${TAG} --name "${CLUSTER}"
  kind load docker-image ${IMG_WS}:${TAG}  --name "${CLUSTER}"
  kind load docker-image ${IMG_UI}:${TAG}  --name "${CLUSTER}"
}

gen_values(){
  FILE="helm/umbrella/values.smoke-kind-ci.yaml"
  step "Generate ${FILE}"
  cat > "${FILE}" <<EOF
namespace: ${NS}

controlPlane:
  enabled: true
  imu-control-plane:
    namespace: ${NS}
    images:
      api: { repository: ${IMG_API}, tag: ${TAG}, pullPolicy: IfNotPresent }
      ws:  { repository: ${IMG_WS},  tag: ${TAG}, pullPolicy: IfNotPresent }
      ui:  { repository: ${IMG_UI},  tag: ${TAG}, pullPolicy: IfNotPresent }
    ingress: { enabled: false }    # אין צורך ב-Ingress בסמוק
    synthetics: { enabled: true, vus: 5, duration: "15s", p95_ms: 900, error_rate: 0.05 }

monitoring: { enabled: true }
gatekeeper:  { enabled: true }
loki:        { enabled: false }    # CI מהיר

externalDNS: { enabled: false }    # לא רוצים תלות בענן בסמוק
ingressNginx: { enabled: true }
certManager:  { enabled: false }

# dashboards בקונטרול-פליין ייטענו כרגיל
EOF
  echo "${FILE}"
}

deploy(){
  local vf="$1"
  step "helm deps build umbrella"
  helm dependency build helm/umbrella

  step "helm upgrade --install ${REL}"
  helm upgrade --install "${REL}" helm/umbrella -n "${NS}" --create-namespace \
    -f helm/umbrella/values.yaml -f helm/umbrella/values.dev.yaml -f "${vf}"
}

tests(){
  step "Wait for control-plane API"
  kubectl -n "${NS}" rollout status deploy/${REL}-control-plane-imu-control-plane-api --timeout=180s || true

  step "helm test control-plane (subchart)"
  # נריץ בדיקות ישירות על צ'ארט המשנה – ה-hooks יפעלו דרך השחרור של umbrella
  set +e
  helm test "${REL}-control-plane-imu-control-plane" -n "${NS}"
  RC=$?
  set -e
  if [[ $RC -ne 0 ]]; then
    echo "Helm test FAILED – rollback umbrella"
    helm rollback "${REL}" 1 -n "${NS}" || true
    exit 1
  fi
  echo "OK"
}

main(){
  build_images
  kind_load
  VF=$(gen_values)
  deploy "$VF"
  tests
}
main "$@"