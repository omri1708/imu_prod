# engine/prebuild/adapter_builder.py
from __future__ import annotations
import os, json, time, hashlib, subprocess, shlex
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    # עדיפות למחסן CAS קיים אצלך
    from provenance.castore import ContentAddressableStore as CAStore
except Exception:
    # Fallback מינימלי
    class CAStore:
        def __init__(self, root: str):
            self.root = root; os.makedirs(root, exist_ok=True)
        def put(self, blob: bytes) -> str:
            h = hashlib.sha256(blob).hexdigest()
            p = os.path.join(self.root, h)
            if not os.path.exists(p):
                with open(p, 'wb') as f: f.write(blob)
            return h

@dataclass
class AdapterArtifact:
    path: str
    digest: str

@dataclass
class AdapterBuildResult:
    ok: bool
    kind: str
    artifacts: List[AdapterArtifact]
    claims: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    logs: List[str]

# ---- capabilities registry signals ----
from engine.capabilities.registry import CapabilityRegistry, CapabilityNeed

REG = CapabilityRegistry(path="var/registry/capabilities.json")
CAS = CAStore("var/cas")

# ---- helper: run shell safely ----
def _run(cmd: List[str], cwd: Optional[str]=None, timeout: int=120) -> Tuple[int,str,str]:
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill(); out, err = p.communicate(); return 124, out, err
    return p.returncode, out, err

# ---- LLM Plug (injected) ----
LLM_FN = None  # יש להזריק פונקציה: fn(prompt:dict)->dict

def set_llm(fn):
    global LLM_FN; LLM_FN = fn

# ---- Synth minimal adapter scaffold ----
TEMPLATES = {
    "android.bundle": {
        "filename": "adapters/android_bundle.py",
        "content": """
from __future__ import annotations
import subprocess, sys
from typing import Dict, Any

def build(job: Dict[str,Any], user: str, workspace: str, policy, ev_index):
    # dry-run בלבד: מדמה קריאה ל-Gradle ולא מבצע דיפלוי
    cmd = ["bash","-lc","echo DRYRUN: ./gradlew bundleRelease && exit 0"]
    p = subprocess.Popen(cmd, cwd=workspace)
    rc = p.wait()
    return {
        "artifacts": {"android.dryrun":"ok"},
        "claims": [{"step":"adapter:android.bundle","ok": rc==0}],
        "evidence": [{"source":"local://android.bundle","trust":0.9,"payload":{"dryrun":True}}]
    }
""".strip()
    },
    "k8s.apply": {
        "filename": "adapters/k8s_apply.py",
        "content": """
from __future__ import annotations
import subprocess
from typing import Dict, Any

def build(job: Dict[str,Any], user: str, workspace: str, policy, ev_index):
    cmd = ["bash","-lc","echo DRYRUN: kubectl apply -f manifest.yaml && exit 0"]
    rc = subprocess.call(cmd, cwd=workspace)
    return {
        "artifacts": {"k8s.dryrun":"ok"},
        "claims": [{"step":"adapter:k8s.apply","ok": rc==0}],
        "evidence": [{"source":"local://k8s.apply","trust":0.9,"payload":{"dryrun":True}}]
    }
""".strip()
    }
}


def synthesize_min_adapter(kind: str, *, workspace: str, policy: Dict[str,Any]) -> Tuple[str, List[str]]:
    """מייצר קובץ/ים מינימליים ל-adapter חסר (dry-run בלבד)."""
    logs: List[str] = []
    os.makedirs(workspace, exist_ok=True)
    tpl = TEMPLATES.get(kind)
    if not tpl and LLM_FN:
        # נסה לייצר תבנית באמצעות LLM לפי מדיניות (חייב לסמן DRYRUN בלבד)
        prompt = {
            "role":"system",
            "policy": policy,
            "task": f"create python adapter stub for '{kind}' with DRYRUN only",
        }
        rsp = LLM_FN(prompt)  # מצפה לשדות {filename, content}
        filename = rsp.get("filename","adapters/auto_stub.py")
        content  = rsp.get("content","# generated stub (dryrun)\n")
        tpl = {"filename": filename, "content": content}
    if not tpl:
        raise RuntimeError(f"no_template_for_kind:{kind}")

    target_path = os.path.join(workspace, tpl["filename"])  # בתוך repo
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(tpl["content"])
    logs.append(f"adapter_stub_written:{target_path}")
    
    from engine.capabilities.pipeline import capability_adoption_flow
    adopt = capability_adoption_flow(kind, proposal={"filename": target_path}, policy=policy)
    if not adopt.get("ok"):
        return AdapterBuildResult(ok=False, kind=kind, artifacts=[], claims=[], evidence=[{"source":"capability.adoption","payload":adopt}], logs=logs)
    
    return target_path, logs



def _lint(path: str) -> Tuple[bool,str]:
    rc, out, err = _run(["bash","-lc", f"python -m pyflakes {shlex.quote(path)} || true"])
    ok = True  # לא חוסם — רק מדווח
    return ok, out+err

def _compile(path: str) -> Tuple[bool,str]:
    rc, out, err = _run(["bash","-lc", f"python -m py_compile {shlex.quote(path)} 2>&1 || true"])
    ok = (rc == 0)
    return ok, out+err

def _dryrun(path: str, workspace: str) -> Tuple[bool,str]:
    # מריץ את הפונקציה build(...) במודול, אבל מצפה ל-DRYRUN בלבד
    code = f"from importlib import import_module; m=import_module('{path.replace('/', '.')[:-3]}');\n" \
           f"print(m.build({{'kind':'stub'}}, 'user','{workspace}', None, None))"  # type: ignore
    rc, out, err = _run(["bash","-lc", f"python - <<'PY'\n{code}\nPY"], cwd=os.getcwd())
    ok = (rc == 0)
    return ok, out+err

def promote_adapter(path: str, allow_live: bool) -> None:
    """אם המדיניות מאפשרת, מקדם את ה-stub ל-LIVE ע"י החלפת DRYRUN בפקודה אמיתית."""
    if not allow_live:
        return
    src = open(path, "r", encoding="utf-8").read()
    src = src.replace("echo DRYRUN:", "")  # פשטני — התאמה לפי kind אפשרית בהמשך
    open(path, "w", encoding="utf-8").write(src)


def build_missing_adapter(kind: str, *, workspace: str, policy: Dict[str,Any]) -> AdapterBuildResult:
    path, logs = synthesize_min_adapter(kind, workspace=workspace, policy=policy)
    promote_adapter(path, allow_live=bool(policy.get("capabilities",{}).get("allow_live", False)))
    
    l_ok, l_out = _lint(path)
    logs.append(l_out)
    c_ok, c_out = _compile(path)
    logs.append(c_out)
    d_ok, d_out = _dryrun(path, workspace)
    logs.append(d_out)

    # evidence + claims
    art_bytes = open(path, 'rb').read()
    digest = CAS.put(art_bytes)
    artifacts = [AdapterArtifact(path=path, digest=digest)]
    claims = [{"t":"adapter_built","kind": kind, "digest": digest, "ok": all([c_ok, d_ok])}]
    evidence = [{"source":"local://adapter_builder","trust":0.95,"ttl_s":86400,
                 "payload":{"path": path, "lint": l_ok, "compile": c_ok, "dryrun": d_ok}}]

    return AdapterBuildResult(ok=bool(d_ok), kind=kind, artifacts=artifacts, claims=claims, evidence=evidence, logs=logs)


def ensure_capabilities(spec: Any, ctx: Dict[str,Any]) -> List[Dict[str,Any]]:
    """בודק אילו יכולות חסרות לפי ה-Registry/Spec, ובונה Stubs דרושים."""
    user = ctx.get("user_id","anon")
    policy = ctx.get("__policy__", {})
    workspace = ctx.get("workspace") or os.getcwd()

    needs: List[CapabilityNeed] = REG.detect_missing(spec, ctx)
    built: List[Dict[str,Any]] = []
    for need in needs:
        res = build_missing_adapter(need.kind, workspace=workspace, policy=policy)
        REG.register_built(need, res)
        built.append({"kind": need.kind, "ok": res.ok, "artifacts": [asdict(a) for a in res.artifacts]})
    return built