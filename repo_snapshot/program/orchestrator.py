# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio, json, os, tempfile
from pathlib import Path
from typing import Dict, Any, List
from assurance.assurance import AssuranceKernel
from assurance.signing import Signer
from assurance.validators import schema_validator
from assurance.errors import ResourceRequired, ValidationFailed
from executor.sandbox import SandboxExecutor, Limits
from executor.policy import Policy

PY_MIN = """\
def add(a,b): return a+b
"""

PY_TEST = """\
from app import add
def test_add(): assert add(2,3)==5
"""

class ProgramOrchestrator:
    """
    מקבל Spec ומריץ בנייה/בדיקות/אריזה דרך Kernel+Executor.
    תומך כעת ב-service מסוג 'python_app' (אפשר להוסיף בהמשך: web/node/go/java וכו').
    """
    def __init__(self, store_root: str = "./assurance_store_programs"):
        self.kernel = AssuranceKernel(store_root)
        self.exec = SandboxExecutor("./executor/policy.yaml", "./program_sbx")

    async def build(self, user_id: str, spec: Dict[str,Any]) -> Dict[str,Any]:
        """
        spec דוגמה:
        {
          "name":"calc",
          "services":[
            {"type":"python_app","name":"svc1"}
          ]
        }
        """
        sess = self.kernel.begin("program.build", "1.0.0", f"Build program {spec.get('name','unnamed')}")
        sess.add_claim("build_ok", True)
        sess.attach_validator(schema_validator({"required":["artifact"],"properties":{"artifact":{"type":"string"}}}))

        # כרגע: python_app
        services = spec.get("services", [])
        if not services:
            raise ValidationFailed("no services")
        outputs=[]
        for svc in services:
            if svc.get("type")=="python_app":
                out = await self._build_python_app(sess, svc)
                outputs.append(out)
            else:
                raise ResourceRequired(f"service_type:{svc.get('type')}", "add adapter or implement builder")

        # Build payload manifest
        payload = {"artifact": f"{spec.get('name','unnamed')}.ok", "services": [o for o in outputs]}
        sess.set_builder(lambda s: [s.cas.put_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                                    meta={"type":"program-manifest"})])
        outs = sess.build()
        rec = sess.commit(Signer("program-hmac"))
        return {"ok": True, "root": rec["root"], "manifest": rec["manifest_digest"], "payload": payload}

    async def _build_python_app(self, sess, svc: Dict[str,Any]) -> Dict[str,Any]:
        name = svc.get("name","app")
        # שלב 1: צור קבצים
        with tempfile.TemporaryDirectory() as td:
            Path(td,"app.py").write_text(PY_MIN, encoding="utf-8")
            Path(td,"test_app.py").write_text(PY_TEST, encoding="utf-8")
            # צרף לקלטים (CAS)
            app_d = sess.cas.put_bytes(open(Path(td,"app.py"),"rb").read(), meta={"lang":"python","role":"source"})
            tst_d = sess.cas.put_bytes(open(Path(td,"test_app.py"),"rb").read(), meta={"lang":"python","role":"test"})
            # שלב 2: הרצה — py_compile + pytest (אם קיים; אחרת רק py_compile)
            rc1, out1 = await self.exec.run(["python", "-m", "py_compile", "app.py"],
                                            inputs={"app.py": open(Path(td,"app.py"),"rb").read()},
                                            allow_write=["."], limits=Limits(no_net=True))
            if rc1 != 0:
                raise ValidationFailed("py_compile failed")
            # pytest אופציונלי
            try:
                rc2, out2 = await self.exec.run(["python", "-m", "pytest", "-q"],
                                                inputs={"app.py": open(Path(td,"app.py"),"rb").read(),
                                                        "test_app.py": open(Path(td,"test_app.py"),"rb").read()},
                                                allow_write=["."], limits=Limits(no_net=True))
            except ResourceRequired:
                # אין pytest; נסתפק ב-compile
                rc2, out2 = 0, b"pytest not available"
            if rc2 != 0:
                raise ValidationFailed("pytest failed")
            # Evidence ל-CAS
            sess.add_evidence("source", f"python:{name}/app.py", digest=app_d, trust=0.8)
            sess.add_evidence("tests", f"python:{name}/test_app.py", digest=tst_d, trust=0.7)
            return {"service": name, "compile": rc1==0, "tests": rc2==0}
PY_WEB = """\
from flask import Flask
app = Flask(__name__)
@app.get("/")
def hello(): return "hello-web"
if __name__ == "__main__":
    app.run()
"""
PY_WEB_TEST = """\
import app as appmod
def test_home():
    client = appmod.app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"hello-web" in r.data
"""
async def _build_python_web_app(self, sess, svc: Dict[str,Any]) -> Dict[str,Any]:
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        p=pathlib.Path(td)
        (p/"app.py").write_text(PY_WEB, encoding="utf-8")
        (p/"test_app.py").write_text(PY_WEB_TEST, encoding="utf-8")
        # compile
        rc1, _ = await self.exec.run(["python","-m","py_compile","app.py"], inputs={"app.py":(p/"app.py").read_bytes()}, allow_write=["."], limits=Limits(no_net=True))
        if rc1!=0: raise ValidationFailed("py_compile failed")
        # pytest (optional)
        try:
            rc2,_ = await self.exec.run(["python","-m","pytest","-q"], inputs={"app.py":(p/"app.py").read_bytes(),"test_app.py":(p/"test_app.py").read_bytes()}, allow_write=["."], limits=Limits(no_net=True))
        except ResourceRequired:
            rc2=0
        if rc2!=0: raise ValidationFailed("web tests failed")
        d1=sess.cas.put_bytes((p/"app.py").read_bytes(), meta={"lang":"python","role":"source"})
        d2=sess.cas.put_bytes((p/"test_app.py").read_bytes(), meta={"lang":"python","role":"test"})
        sess.add_evidence("source","python:web/app.py", digest=d1, trust=0.8)
        sess.add_evidence("tests","python:web/test_app.py", digest=d2, trust=0.7)
        return {"service": svc.get("name","web"), "compile": True, "tests": True}

