# tools/imu_doctor.py
import os, sys, json, ast, shutil, importlib.util
from pathlib import Path

ROOT = Path(os.getcwd())

def _find(patterns, exts=(".py",)):
    files = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in exts:
            try:
                txt = p.read_text("utf-8", errors="ignore")
            except Exception:
                continue
            if any(x in txt for x in patterns):
                files.append(str(p.relative_to(ROOT)))
    return files

def _ast_find_fastapi_apps():
    res=[]
    for p in ROOT.rglob("*.py"):
        try:
            tree=ast.parse(p.read_text("utf-8",errors="ignore"))
        except Exception: 
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for t in [ast.Name, ast.Attribute]:
                    if isinstance(n.value, ast.Call) and isinstance(n.value.func, t):
                        name = getattr(n.value.func,"id",None) or getattr(n.value.func,"attr",None)
                        if str(name)=="FastAPI":
                            for t in n.targets:
                                if isinstance(t,ast.Name):
                                    res.append((str(p.relative_to(ROOT)), t.id))
    return res

def _ast_find_include_router():
    res=[]
    for p in ROOT.rglob("*.py"):
        try:
            tree=ast.parse(p.read_text("utf-8",errors="ignore"))
        except Exception:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fn = n.func
                if isinstance(fn, ast.Attribute) and fn.attr=="include_router":
                    try:
                        target = ast.unparse(fn.value)
                        arg = ast.unparse(n.args[0]) if n.args else ""
                    except Exception:
                        target = "<app>"
                        arg = "<router>"
                    res.append((str(p.relative_to(ROOT)), target, arg))
    return res

def _registry_map():
    try:
        import importlib.util, importlib.machinery
        sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location("reg", ROOT/"engine/blueprints/registry.py")
        reg = importlib.util.module_from_spec(spec); spec.loader.exec_module(reg)  # type: ignore
        mp = {}
        for k,v in getattr(reg,"_REGISTRY",{}).items():
            mp[k] = f"{v.__module__}.{getattr(v,'__name__','')}"
        try:
            r_api = reg.resolve("api")
            mp["api_resolves_to"] = f"{r_api.__module__}.{getattr(r_api,'__name__','')}"
        except Exception as e:
            mp["api_resolves_to"] = f"ERROR:{e}"
        return mp
    except Exception as e:
        return {"error": str(e)}

def _builder_checks():
    f = ROOT/"engine/build_orchestrator.py"
    info={"exists": f.exists()}
    if not f.exists(): return info
    txt = f.read_text("utf-8", errors="ignore")
    info["uses_tmpdir"] = "_build_with_tmpdir" in txt
    info["uses_sandbox"] = "_build_with_sandbox" in txt
    info["pytest_target_logic"] = ("services/api/tests" in txt) or ("pytest_target" in txt)
    return info

def _policy_checks():
    f = ROOT/"executor/policy.yaml"
    info={"exists": f.exists()}
    if not f.exists(): return info
    y = f.read_text("utf-8", errors="ignore")
    info["allows_python"] = "python" in y and "allowed_tools" in y
    return info

def _env_suspects():
    return {
        "bwrap_in_venv": shutil.which("bwrap") or "",
        "sitecustomize_present": (ROOT/"sitecustomize.py").exists(),
    }

def _app_file_snapshot():
    out = []
    for p in ["out/services/api/app.py","out/services/api/tests/test_acceptance_generated.py"]:
        fp = ROOT/p
        if fp.exists():
            out.append({p: (fp.read_text("utf-8",errors="ignore")[:400])})
    return out

def main():
    report = {
        "fastapi_apps": _ast_find_fastapi_apps(),
        "include_router_calls": _ast_find_include_router(),
        "blueprint_registry": _registry_map(),
        "builder": _builder_checks(),
        "policy": _policy_checks(),
        "env_suspects": _env_suspects(),
        "generated_snapshots": _app_file_snapshot(),
        "grep_flask_like": _find(["@app.route","from flask","jsonify"]),
        "grep_fastapi_like": _find(["FastAPI(","from fastapi import"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
