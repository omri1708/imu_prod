set -euo pipefail

# שימוש:
#   NS=dev RELEASE=imu VALUES=helm/control-plane/values.yaml ./scripts/quick_deploy.sh dry
#   NS=dev RELEASE=imu VALUES=helm/control-plane/values.yaml ./scripts/quick_deploy.sh deploy
#
# דרישות: kubectl + helm מותקנים ומכוונים לקלאסטר.

NS="${NS:-default}"
RELEASE="${RELEASE:-imu}"
VALUES="${VALUES:-helm/control-plane/values.yaml}"
CHART="helm/control-plane"

step() { echo -e "\n\033[1;36m==> $*\033[0m"; }

if [[ "${1:-}" == "dry" ]]; then
  step "helm template (gating expected to pass)"
  helm template "$RELEASE" "$CHART" -n "$NS" -f "$VALUES" >/tmp/cp.yaml
  echo "rendered to /tmp/cp.yaml"
  exit 0
fi

if [[ "${1:-}" == "deploy" ]]; then
  step "helm upgrade --install"
  helm upgrade --install "$RELEASE" "$CHART" -n "$NS" -f "$VALUES" --create-namespace

  step "kubectl get svc,pods -n $NS"
  kubectl get svc,pods -n "$NS" | sed -n '1,50p'

  step "helm test $RELEASE -n $NS (Chart hooks ירוצו גם PostSync)"
  set +e
  helm test "$RELEASE" -n "$NS"
  TEST_RC=$?
  set -e
  if [[ $TEST_RC -ne 0 ]]; then
    echo "helm test FAILED, performing rollback"
    helm rollback "$RELEASE" 1 -n "$NS" || true
    exit 1
  fi
  echo "OK"
  exit 0
fi

echo "usage: quick_deploy.sh [dry|deploy]"
exit 2