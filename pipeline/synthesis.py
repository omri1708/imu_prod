# imu_repo/pipeline/synthesis.py
from __future__ import annotations
import os, subprocess, tempfile, shutil
from typing import Dict, Any, List

from core.contracts.verifier import Contracts
from grounded.fact_gate import FactGate, EvidenceIndex, SchemaRule, UnitRule, FreshnessRule
from provenance.store import ProvenanceStore
from grounded.audit import AuditLog

class SynthesisError(Exception): ...

class SynthesisPipeline:
    """Full synthesis pipeline: plan → generate → test → verify → package"""

    def __init__(self,root:str=".imu_state/synthesis"):
        self.root=root
        os.makedirs(self.root,exist_ok=True)
        self.contracts=Contracts()
        self.prov=ProvenanceStore(os.path.join(self.root,"prov"))
        self.fact_gate=FactGate(EvidenceIndex(self.prov), rules=[SchemaRule(),UnitRule(),FreshnessRule()])
        self.audit=AuditLog(os.path.join(self.root,"audit.jsonl"))

    def plan(self,req:str)->Dict[str,Any]:
        self.audit.append("plan",{"req":req})
        # תכנון דטרמיניסטי פשוט
        return {"components":["moduleA","moduleB"],"req":req}

    def generate(self,plan:Dict[str,Any])->List[str]:
        tmpdir=tempfile.mkdtemp(prefix="gen_",dir=self.root)
        files=[]
        for comp in plan["components"]:
            path=os.path.join(tmpdir,f"{comp}.py")
            with open(path,"w") as f:
                f.write(f"# code for {comp}\nprint('{comp} OK')\n")
            files.append(path)
        self.audit.append("generate",{"files":files})
        return files

    def test(self,files:List[str])->None:
        for f in files:
            res=subprocess.run(["python3",f],capture_output=True,text=True)
            if res.returncode!=0:
                self.audit.append("test_fail",{"file":f,"err":res.stderr})
                raise SynthesisError(f"test_failed:{f}")
        self.audit.append("test_pass",{"files":files})

    def verify(self,files:List[str])->None:
        for f in files:
            claim=f"file:{os.path.basename(f)}"
            prov_id=self.prov.add(claim,sources=["local"],payload={"verified":True})
            ok, diags = self.fact_gate.check_claims([{"claim":claim,"sources":["local"]}], strict=True)
            if not ok:
                raise SynthesisError(f"verify_failed:{f}; diags={diags}")
        self.audit.append("verify",{"files":files})

    def package(self,files:List[str])->str:
        pkg=os.path.join(self.root,"package")
        os.makedirs(pkg,exist_ok=True)
        for f in files:
            shutil.copy(f,pkg)
        self.audit.append("package",{"path":pkg})
        return pkg

    def run(self,req:str)->str:
        plan=self.plan(req)
        files=self.generate(plan)
        self.test(files)
        self.verify(files)
        return self.package(files)
