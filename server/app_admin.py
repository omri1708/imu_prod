# server/app_admin.py
from fastapi import FastAPI, Depends
from server.deps.admin_auth import require_admin

# Routers אדמין (לפי הרשימה שהופיעה ב-include_router_calls של http_api) :contentReference[oaicite:4]{index=4}
from server.emergency_api import router as emergency_router
from server.controlplane_deploy_api import router as cp_deploy_router
from server.helm_template_synth_api import router as helm_synth_router
from server.k8s_template_synth_api import router as k8s_synth_router
from server.synth_presets_api import router as synth_presets_router
from server.synth_adapter_api import router as synth_router
from server.metrics_jobs_api import router as jobs_metrics_router
from server.gh_status_api import router as gh_status_router
from server.canary_auto_api import router as auto_canary_router
from server.canary_controller import router as canary_router
from server.merge_guard_api import router as merge_guard_router
from server.gatekeeper_api import router as gatekeeper_router
from server.webhooks_api import router as webhooks_router
from server.policy_edit_api import router as policy_edit_router
from server.gitops_checks_api import router as gh_checks_router
from server.gitops_guard_api import router as guard_router
from server.gitops_api import router as gitops_router
from server.replay_api import router as replay_router
from server.unified_archive_api import router as unified_router
from server.bundles_api import router as bundles_router
from server.archive_api import router as archive_router
from server.key_admin_api import router as key_admin_router
from server.provenance_api import router as prov_router
from server.metrics_api import router as metrics_router
from server.supplychain_api import router as supply_router
from server.events_api import router as events_router  # אם לא קיים, מחק שורה זו
from server.supplychain_index_api import router as sc_index_router
from server.runbook_api import router as runbook_router

APP = FastAPI(title="IMU Admin API", version="0.0.1")

# כל ראוטר אדמין—מאובטח בדיפולט
def secure_include(r):
    APP.include_router(r, dependencies=[Depends(require_admin)])

for r in [
    emergency_router, cp_deploy_router, helm_synth_router, k8s_synth_router,
    synth_presets_router, synth_router, jobs_metrics_router, gh_status_router,
    auto_canary_router, canary_router, merge_guard_router, gatekeeper_router,
    webhooks_router, policy_edit_router, gh_checks_router, guard_router, gitops_router,
    replay_router, unified_router, bundles_router, archive_router, key_admin_router,
    prov_router, metrics_router, supply_router, sc_index_router, runbook_router
]:
    secure_include(r)
