# scripts/one_button_platform.sh

# One-button deploy: Kind + local registry + Umbrella(dev) (control-plane + monitoring + gatekeeper + loki)
# דרישות: kind, kubectl, helm, docker
set -euo pipefail

CLUSTER="${CLUSTER:-imu}"
NS="${NS:-dev}"
REL="${REL:-imu-umbrella}"
REG_NAME="${REG_NAME:-imu-reg}"
REG_PORT="${REG_PORT:-5001}"

step(){ echo -e "\n\033[1;36m==> $*\033[0m"; }

ensure_tools(){
  command -v kind >/dev/null || { echo "kind missing"; exit 1; }
  command -v kubectl >/dev/null || { echo "kubectl missing"; exit 1; }
  command -v helm >/dev/null || { echo "helm missing"; exit 1; }
  command -v docker >/dev/null || { echo "docker missing"; exit 1; }
}

create_kind(){
  ./scripts/kind_setup.sh || true
}

build_umbrella_deps(){
  step "helm dependency build helm/umbrella"
  helm dependency build helm/umbrella
}

install_umbrella(){
  step "Deploy Umbrella (dev)"
  helm upgrade --install "${REL}" helm/umbrella -n "${NS}" --create-namespace \
    -f helm/umbrella/values.yaml -f helm/umbrella/values.dev.yaml \
    --set monitoring.enabled=true \
    --set gatekeeper.enabled=true \
    --set loki.enabled=true || (echo "helm install failed"; exit 1)
}

wait_core(){
  step "Waiting for core components (up to ~2min)"
  kubectl -n "${NS}" rollout status deploy/"${REL}"-control-plane-imu-control-plane-api --timeout=120s || true
  kubectl -n monitoring rollout status deploy/"${REL}"-kube-prometheus-stack-grafana --timeout=120s || true
  kubectl -n gatekeeper-system rollout status deploy/gatekeeper-controller-manager --timeout=120s || true
}

notes(){
  cat <<EOF

OK 🎉
- Namespace: ${NS}
- Services:   API/WS/UI via ${REL}-control-plane-imu-control-plane-svc.${NS}.svc
- Grafana:    kube-prometheus-stack (monitoring ns). Use port-forward:
    kubectl -n monitoring port-forward svc/${REL}-kube-prometheus-stack-grafana 3000:80
    login: admin / prom-operator (ברירת מחדל צ'ארט)
- Loki:       הופעל (אם loki.enabled=true). Promtail שולח לוגים.

תאחסון/ניקוי:
  helm uninstall ${REL} -n ${NS}
  kind delete cluster --name ${CLUSTER}
EOF
}

main(){
  ensure_tools
  create_kind
  build_umbrella_deps
  install_umbrella
  wait_core
  notes
}
main "$@"