# server/app_public.py
from fastapi import FastAPI, Depends
from server.middleware.evidence import evidence_enforcer
from server.deps.evidence_gate import require_citations_or_silence
from server.deps.sandbox import require_sandbox_ready

# Routers ציבוריים (לפי הצילום שלך קיימים בתיקיית server/routers) :contentReference[oaicite:3]{index=3}
from server.routers.chat_api import router as chat_router
from server.routers.respond_api import router as respond_router
from server.routers.orchestrate_api import router as orchestrate_router
from server.routers.program_api import router as program_router
from server.routers.build_api import router as build_router
from server.routers.prebuild_api import router as prebuild_router
from server.routers.cache_api import router as cache_router
from server.routers.consent_api import router as consent_router
from server.routers.adapters_secure import router as adapters_secure_router

from server.routers.health_api import router as health_router  # נוסיף בסעיף 6

APP = FastAPI(title="IMU Public API", version="0.0.1")
APP.middleware("http")(evidence_enforcer)

# ראוטרים “מחזירי-תוכן”: מוסיפים אכיפת Evidence לפני ריצה
APP.include_router(chat_router,       dependencies=[Depends(require_citations_or_silence)])
APP.include_router(respond_router,    dependencies=[Depends(require_citations_or_silence)])
APP.include_router(orchestrate_router,dependencies=[Depends(require_citations_or_silence),
                                                    Depends(require_sandbox_ready)])
APP.include_router(program_router,    dependencies=[Depends(require_citations_or_silence)])
APP.include_router(build_router,      dependencies=[Depends(require_citations_or_silence),
                                                    Depends(require_sandbox_ready)])

# שאינם מחזירי-תוכן (או עזר)
APP.include_router(prebuild_router)
APP.include_router(cache_router)
APP.include_router(consent_router)
APP.include_router(adapters_secure_router)

APP.include_router(health_router, prefix="/health")
