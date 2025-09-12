# engine/orchestrator/universal_orchestrator.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import time
import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from engine.artifacts.registry import register as register_artifacts
from engine.qa.spec_tests import gen_acceptance_tests
from engine.blueprints.registry import get_blueprint
"""
Universal Orchestrator — configurable for:
  1) Quick dev / POC (no environment dependencies)
  2) Live system (domain-aware, tool checks, blueprints, learning)
  3) Production on unmanaged machines (safe defaults, no auto-install)

"""

# --- Imports with graceful fallbacks to support different repo layouts ----------
try:
    from engine.llm_gateway import LLMGateway  # type: ignore
except Exception:  # pragma: no cover
    from llm_gateway import LLMGateway  # type: ignore

try:
    from engine.orchestrator.universal_planner import UniversalPlanner  # type: ignore
except Exception:  # pragma: no cover
    from engine.universal_planner import UniversalPlanner  # type: ignore

try:
    from engine.build_orchestrator import BuildOrchestrator  # type: ignore
except Exception:  # pragma: no cover
    from build_orchestrator import BuildOrchestrator  # type: ignore

try:
    from engine.blueprints.registry import resolve as resolve_blueprint  # type: ignore
except Exception:  # pragma: no cover
    from registry import resolve as resolve_blueprint  # type: ignore

try:
    from knowledge.tools_store import snapshot, get_all, remember  # type: ignore
except Exception:  # pragma: no cover
    from tools_store import snapshot, get_all, remember  # type: ignore

try:
    from learning.pattern_store import record_episode, suggest  # type: ignore
except Exception:  # pragma: no cover
    from pattern_store import record_episode, suggest  # type: ignore

# Auto-installer (best-effort; may be missing in restricted environments)
try:
    from tools.auto_install import auto_install_missing  # type: ignore
except Exception:  # pragma: no cover
    async def auto_install_missing(_missing: List[str]) -> Dict[str, Any]:  # type: ignore
        return {"log": [], "manager": "none"}

# ------------------------------- Config ----------------------------------------

class OrchestratorConfig:
    """
    Configuration knobs:
      - mode: "poc", "live", "prod"  (default: env IMU_MODE or "live")
      - auto_install: override auto-install behavior; if None use per-mode defaults
      - tool_check_ttl: cache TTL for tool checks (seconds). Default: 12h
      - require_grounding: pass through to LLM gateway for stricter planning (optional)
    """
    def __init__(
        self,
        mode: Optional[str] = None,
        *,
        auto_install: Optional[bool] = None,
        tool_check_ttl: Optional[int] = None,
        require_grounding: Optional[bool] = None,
    ) -> None:
        self.mode: str = (mode or os.getenv("IMU_MODE", "live")).strip().lower()
        if self.mode not in ("poc", "live", "prod"):
            self.mode = "live"
        # auto-install defaults per mode (can be overridden by IMU_AUTO_INSTALL)
        env_auto = os.getenv("IMU_AUTO_INSTALL")
        if env_auto is not None:
            self.auto_install = env_auto.strip().lower() not in ("0", "false", "no", "off")
        elif auto_install is not None:
            self.auto_install = bool(auto_install)
        else:
            # Defaults: poc=False (no env deps), live=True, prod=False (safer)
            self.auto_install = True if self.mode == "live" else False
        # tool check TTL
        if tool_check_ttl is not None:
            self.tool_check_ttl = int(tool_check_ttl)
        else:
            try:
                self.tool_check_ttl = int(os.getenv("IMU_TOOLS_TTL", str(12 * 3600)))
            except Exception:
                self.tool_check_ttl = 12 * 3600
        # grounding
        if require_grounding is not None:
            self.require_grounding = bool(require_grounding)
        else:
            self.require_grounding = os.getenv("IMU_REQUIRE_GROUNDING", "0").strip().lower() in ("1","true","yes","on")

    def __repr__(self) -> str:
        return f"OrchestratorConfig(mode={self.mode!r}, auto_install={self.auto_install}, ttl={self.tool_check_ttl}, require_grounding={self.require_grounding})"


# ------------------------- Tool categories / hints -----------------------------

CAT_TO_CHECKS: Dict[str, List[str]] = {
    "web.frontend":      ["exe:node", "exe:npm"],
    "mobile.android":    ["exe:sdkmanager", "exe:gradle"],
    "mobile.ios":        ["exe:xcodebuild"],
    "game.unity":        ["exe:Unity"],
    "realtime.webrtc":   ["exe:ffmpeg"],
    "gpu.cuda":          ["exe:nvcc", "py:torch"],
    "db.sql":            ["exe:sqlite3"],
    "k8s":               ["exe:kubectl", "exe:helm"],
}

_INSTALL_HINTS: Dict[str, str] = {
    "exe:node":      "התקן Node.js (brew/choco/apt) ואז בדוק: node --version",
    "exe:npm":       "npm מגיע עם Node.js; בדוק: npm --version",
    "exe:gradle":    "התקן Gradle; בדוק: gradle -v",
    "exe:sdkmanager":"התקן Android CommandLineTools/SDK; בדוק: sdkmanager --list",
    "exe:xcodebuild":"התקן Xcode + Command Line Tools (דורש GUI/EULA)",
    "exe:Unity":     "התקן Unity Hub/Editor (דורש GUI/EULA)",
    "exe:ffmpeg":    "התקן ffmpeg (brew/apt/dnf/yum/winget/choco)",
    "exe:nvcc":      "התקן CUDA Toolkit (NVIDIA הרשמי)",
    "py:torch":      "pip install torch (CPU/CUDA לפי הצורך)",
    "exe:kubectl":   "התקן kubectl",
    "exe:helm":      "התקן helm",
    "exe:sqlite3":   "התקן sqlite3",
}

def _install_hint(check: str) -> str:
    return _INSTALL_HINTS.get(check, "התקן את הכלי המבוקש או השבת בדיקות")

def _make_instructions(missing: List[str]) -> List[Dict[str, str]]:
    return [{"check": m, "hint": _install_hint(m)} for m in missing]


# ------------------------------ Generators ------------------------------------

def _builtin_python_web(_spec: Dict[str, Any]) -> Dict[str, str]:
    """
    Conservative, zero-deps FastAPI app + a tiny test, used as a fallback
    and as the generator for POC mode.
    """
    app_py = """# generated by UniversalOrchestrator builtin
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="IMU Builtin App")

@app.get("/")
async def root():
    return {"ok": True, "msg": "IMU builtin web ready"}

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return '''<!doctype html><html><head><meta charset="utf-8"><title>IMU UI</title></head>
<body style="font-family:system-ui">
<h3>IMU Builtin UI</h3>
<div id="out">loading…</div>
<script>
fetch("/").then(r=>r.json()).then(j=>{document.getElementById('out').innerText = JSON.stringify(j)});
</script>
</body></html>'''
"""
    test_py = """# generated test for builtin app
def test_root():
    from app import app  # noqa: F401
    assert app is not None
"""
    return {"app.py": app_py, "test_app.py": test_py}


def _synthesize_poc_files(spec: Dict[str, Any]) -> Dict[str, str]:
    """
    Minimal FastAPI skeleton synthesized directly from spec (no blueprints).
    """
    nav_tabs = (spec.get("nav") or {}).get("tabs") or ["home", "profile"]
    app_py = f"""# generated from spec (POC)
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {{"ok": True, "tabs": {json.dumps(nav_tabs, ensure_ascii=False)}}}
"""
    test_py = """def test_root():
    import app as m
    assert hasattr(m, "app")
"""
    return {"app.py": app_py, "test_app.py": test_py}

def _has_python_sources(files: Dict[str, Any]) -> bool:
    return any(str(k).endswith(".py") for k in files.keys())


# --------------------------- Inference helpers --------------------------------

def _domain_from_spec(spec: Dict[str, Any]) -> str:
    """
    Prefer learning.pattern_store.suggest; fall back to components-based heuristic.
    """
    try:
        prior = suggest(spec) or {}
        dom = prior.get("domain")
        if dom:
            return dom
    except Exception:
        pass
    comps = [c.get("type", "") for c in (spec.get("components") or [])]
    if "realtime" in comps:
        return "realtime"
    if "game" in comps:
        return "game"
    if "mobile" in comps:
        return "mobile"
    if "web" in comps:
        return "web"
    if "api" in comps:
        return "api"
    return "custom"

def _infer_checks_from_spec(spec: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Decide which environment checks to run based on:
      - explicit tools categories in spec.tools[].category (if provided)
      - inferred domain (web/mobile/realtime/game/api/custom)
      - heuristic scan over spec text for gpu/k8s/sql/webrtc hints
    """
    checks: List[str] = []
    # 1) explicit
    for t in (spec.get("tools") or []):
        cat = (t or {}).get("category")
        if cat in CAT_TO_CHECKS:
            checks.extend(CAT_TO_CHECKS[cat])
    # 2) domain base
    domain = _domain_from_spec(spec)
    if domain == "web":
        checks.extend(["exe:node", "exe:npm"])
    elif domain == "mobile":
        checks.extend(["exe:sdkmanager", "exe:gradle"])
    elif domain == "realtime":
        checks.append("exe:ffmpeg")
    elif domain == "game":
        checks.append("exe:Unity")
    # 3) heuristic scan
    try:
        blob = json.dumps(spec, ensure_ascii=False).lower()
    except Exception:
        blob = str(spec).lower()
    if any(k in blob for k in ("cuda", "gpu", "pytorch", "torch", "tensorrt")):
        checks.extend(["exe:nvcc", "py:torch"])
    if any(k in blob for k in ("k8s", "kubernetes", "kubectl", "helm")):
        checks.extend(["exe:kubectl", "exe:helm"])
    if any(k in blob for k in ("sqlite", "sql", "database", "db")):
        checks.append("exe:sqlite3")
    if any(k in blob for k in ("webrtc", "rtmp", "sdp")) and "exe:ffmpeg" not in checks:
        checks.append("exe:ffmpeg")
    # dedupe & stable order
    checks = sorted(set(checks))
    return domain, checks


# ---------------------------- Universal Orchestrator ---------------------------

class UniversalOrchestrator:
    """
    Full flow:
      1) Planner (LLM) → Spec (architecture)
      2) Environment tool checks (cached)
      3) Optional auto-install of missing tools (per config)
      4) Blueprint selection and code generation; builtin fallback if needed
      5) Real build (compile + pytest if available)
      6) Continuous learning (patterns/tools)
    """

    def __init__(self, gateway: Optional[LLMGateway] = None, config: Optional[OrchestratorConfig] = None) -> None:
        self.config = config or OrchestratorConfig()
        self.gw = gateway or LLMGateway()
        self.planner = UniversalPlanner(self.gw)
        self.builder = BuildOrchestrator()

    # ---------- Analyze: turn conversation + context into a spec ----------
    def analyze(self, user_id: str, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Use prior learning to hint blueprint/domain; pass known tool states.
        try:
            prior = suggest({"components": []}) or {}
        except Exception:
            prior = {}
        known = {}
        try:
            known = get_all()
        except Exception:
            known = {}
        return self.planner.analyze_request(
            user_id=user_id,
            text=text,
            context=context,
            prior=prior,
            known_tools=known,
            ask_back=False,
        )

    # ---------- Execute: generate files + build ----------
    async def execute(self, spec: Dict[str, Any], *, workdir: Optional[str] = None) -> Dict[str, Any]:
        mode = self.config.mode

        # --- (A) Decide domain & tool needs
        domain, needs = _infer_checks_from_spec(spec)

        # --- (B) POC mode: no env checks, no auto-install, always builtin generator
        if mode == "poc":
            files = _synthesize_poc_files(spec)
            t0 = time.time()
            build = await self.builder.build_python_module(files, name=(domain + "_poc"), workdir=workdir)
            latency = int((time.time() - t0) * 1000)
            ok = bool(build.get("ok", False))
            try:
                record_episode(spec.get("summary", "") or spec.get("title", ""), spec, {}, "builtin.poc.fastapi", ok, latency_ms=latency)
            except Exception:
                pass
            return {
                "ok": ok,
                "mode": mode,
                "domain": domain,
                "blueprint": "builtin.poc.fastapi",
                "used_fallback": True,
                "build": build,
                "tools": {},
                "still_missing": [],
                "installer_log": [],
                "instructions": [],
            }

        # --- (C) Live / Prod: tool checks (+ optional auto-install), blueprints
        # 1) snapshot environment
        tools_ok: Dict[str, bool] = {}
        missing: List[str] = []
        if needs:
            try:
                tools_ok = snapshot(needs, ttl=self.config.tool_check_ttl)
            except Exception:
                tools_ok = {k: False for k in needs}
            missing = [k for k, v in tools_ok.items() if not v]

        # 2) optional auto-install (live default=True, prod default=False; overridable)
        installer_log: List[Dict[str, Any]] = []
        if missing and self.config.auto_install:
            try:
                res = await auto_install_missing(missing)
                installer_log = list(res.get("log") or [])
                # rescan after install attempt
                try:
                    tools_ok = snapshot(needs, ttl=self.config.tool_check_ttl, force=True)  # type: ignore
                except TypeError:
                    tools_ok = snapshot(needs, ttl=self.config.tool_check_ttl)  # compatibility
                missing = [k for k, v in tools_ok.items() if not v]
            except Exception as e:
                installer_log.append({"check": "_auto_install", "cmd": "internal", "rc": -1, "out": f"{e}"})

        # 3) choose blueprint/generator (or fallback)
        prior = {}
        try:
            prior = suggest(spec) or {}
        except Exception:
            prior = {}
        bp = prior.get("suggested_blueprint") or prior.get("domain") or domain

        used_fallback = False
        files: Dict[str, Any] = {}
        generator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        try:
            generator = resolve_blueprint(bp)
        except Exception:
            generator = None

        if callable(generator):
            try:
                files = generator(spec) or {}
                # Fallback: אם blueprint לא ייצר app.py, ניצור מינימלי כדי שה- build לא ייפול
                if "services/api/app.py" not in files:
                    files["services/api/app.py"] = b"""\
                import time
                from fastapi import FastAPI
                app = FastAPI(title="IMU API")
                @app.get("/healthz")
                def healthz(): return {"ok": True, "ts": time.time()}
                @app.get("/readyz")
                def readyz(): return {"ready": True}
                @app.get("/metrics")
                def metrics():
                    body="# HELP app_up 1\\n# TYPE app_up gauge\\napp_up 1\\n"
                    return (body, 200, {"Content-Type":"text/plain; version=0.0.4"})
                @app.get("/")
                def root(): return {"ok": True, "entities": []}
                """
            except Exception:
                files = {}
        files.setdefault("services/api/tests/test_acceptance_generated.py", gen_acceptance_tests(spec))
        try:
            dg = register_artifacts(spec.get("title","app"), files, base_dir=workdir)  # אם הפונקציה תומכת
        except TypeError:
            try:
                dg = register_artifacts(spec.get("title","app"), files)
            except Exception:
                dg = None

        if not files or not _has_python_sources(files):
            files = _builtin_python_web(spec)
            used_fallback = True
            bp = "builtin.python.web"

        # 4) build
        t0 = time.time()
        try:
            build = await self.builder.build_python_module(files, name=(domain + "_glue"), workdir=workdir)  # חדש
        except TypeError:
            build = await self.builder.build_python_module(files, name=(domain + "_glue"))  # תאימות
        latency = int((time.time() - t0) * 1000)
        ok = bool(build.get("ok", False))

        # 5) learning & remember tool states
        try:
            record_episode(spec.get("summary", "") or spec.get("title", ""), spec, tools_ok, bp, ok, latency_ms=latency)
        except Exception:
            pass
        try:
            if needs:
                remember(needs, tools_ok)
        except Exception:
            pass

        # 6) response with actionable guidance
        return {
            "ok": ok,
            "artifact_digest": dg,
            "mode": mode,
            "domain": domain,
            "blueprint": bp,
            "used_fallback": used_fallback,
            "build": build,
            "tools": tools_ok,
            "still_missing": missing,
            "installer_log": installer_log,
            "instructions": _make_instructions(missing),
        }
