# scripts/gen_alerts_values.sh
# מייצר קובץ values.overlay.yaml (לא נשמר בגיט) עם ערכי alerts.* ומדביק Helm.
# שימוש:
#   ./scripts/gen_alerts_values.sh --env prod \
#     --slack $SLACK_WEBHOOK --teams $TEAMS_WEBHOOK \
#     --email-to oncall@example.com --email-from alerts@example.com \
#     --smtp smtp.example.com:587 --smtp-user user --smtp-pass pass
set -euo pipefail

ENV="dev"
OUT="values.alerts.overlay.yaml"
SLACK=""; TEAMS=""; EMAIL_TO=""; EMAIL_FROM=""; SMTP=""; SMTP_USER=""; SMTP_PASS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --slack) SLACK="$2"; shift 2;;
    --teams) TEAMS="$2"; shift 2;;
    --email-to) EMAIL_TO="$2"; shift 2;;
    --email-from) EMAIL_FROM="$2"; shift 2;;
    --smtp) SMTP="$2"; shift 2;;
    --smtp-user) SMTP_USER="$2"; shift 2;;
    --smtp-pass) SMTP_PASS="$2"; shift 2;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done

cat > "${OUT}" <<EOF
alerts:
  slack: { webhook: "${SLACK}" }
  teams: { webhook: "${TEAMS}" }
  email:
    to: "${EMAIL_TO}"
    from: "${EMAIL_FROM}"
    smarthost: "${SMTP}"
    authUsername: "${SMTP_USER}"
    authPassword: "${SMTP_PASS}"
EOF

echo "generated ${OUT} (env=${ENV})"
echo "Example Helm:"
echo "  helm upgrade --install umbrella helm/umbrella -n ${ENV} -f helm/umbrella/values.yaml -f helm/umbrella/values.${ENV}.yaml -f ${OUT}"