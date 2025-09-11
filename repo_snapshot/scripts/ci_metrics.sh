# scripts/ci_metrics.sh
# Helper: push metrics/logs to Pushgateway/Loki from CI steps.
# Usage:
#   PG_URL=$PUSHGATEWAY_URL loki_url=$LOKI_URL ./scripts/ci_metrics.sh metric <job> <labels key=val,...> <name> <value>
#   loki_url=$LOKI_URL ./scripts/ci_metrics.sh log <stream key=val,...> <MESSAGE_MULTILINE...>
set -euo pipefail

PG_URL="${PG_URL:-${PUSHGATEWAY_URL:-}}"
LOKI_URL="${LOKI_URL:-${loki_url:-}}"

metric() {
  local job="$1"; shift
  local labels="$1"; shift
  local name="$1"; shift
  local value="$1"; shift
  [[ -z "$PG_URL" ]] && { echo "Pushgateway URL not set, skipping metric"; return 0; }
  # labels: key=val,key2=val2
  local path="metrics/job/${job}"
  IFS=',' read -r -a arr <<< "$labels"
  for kv in "${arr[@]}"; do
    k="${kv%%=*}"; v="${kv#*=}"
    path="${path}/${k}/${v}"
  done
  cat <<EOF | curl -s --data-binary @- "${PG_URL}/${path}"
# TYPE ${name} gauge
${name}${labels:+{${labels//,/ , }}} ${value}
EOF
}

log() {
  local stream="$1"; shift
  [[ -z "$LOKI_URL" ]] && { echo "Loki URL not set, skipping log"; return 0; }
  local msg="$*"
  local ts=$(date +%s%N)
  # stream labels: key=val,key2=val2
  local json_labels="{"
  IFS=',' read -r -a arr <<< "$stream"
  for kv in "${arr[@]}"; do
    k="${kv%%=*}"; v="${kv#*=}"
    json_labels="${json_labels}\"${k}\":\"${v}\","
  done
  json_labels="${json_labels}\"source\":\"ci\"}"
  cat > /tmp/loki_ci.json <<EOF
{ "streams": [ { "stream": ${json_labels}, "values": [[ "${ts}", $(printf '%q' "${msg}") ]] } ] }
EOF
  curl -s -H "Content-Type: application/json" -X POST --data-binary @/tmp/loki_ci.json "${LOKI_URL}/loki/api/v1/push" >/dev/null || true
}

case "${1:-}" in
  metric) shift; metric "$@";;
  log)    shift; log "$@";;
  *) echo "usage: $0 metric <job> <labels> <name> <value> | log <stream> <msg...>"; exit 2;;
esac