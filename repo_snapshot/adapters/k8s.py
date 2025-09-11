# adapters/k8s.py
# -*- coding: utf-8 -*-
import json, subprocess, shutil, os, tempfile
from typing import Dict
from policy.policy_engine import PolicyViolation

def dry_run()->bool:
    return shutil.which("kubectl") is not None and shutil.which("helm") is not None

def deploy_image(namespace:str, name:str, image:str, port:int=8080)->Dict:
    if not dry_run(): raise PolicyViolation("k8s tools missing (kubectl/helm)")
    manifest = {
      "apiVersion":"apps/v1","kind":"Deployment",
      "metadata":{"name":name,"namespace":namespace},
      "spec":{"replicas":1,"selector":{"matchLabels":{"app":name}},
        "template":{"metadata":{"labels":{"app":name}},
          "spec":{"containers":[{"name":name,"image":image,"ports":[{"containerPort":port}]}]}}}
    }
    svc = {
      "apiVersion":"v1","kind":"Service",
      "metadata":{"name":f"{name}-svc","namespace":namespace},
      "spec":{"selector":{"app":name},"ports":[{"port":port,"targetPort":port}]}
    }
    with tempfile.TemporaryDirectory() as d:
        dep = os.path.join(d,"dep.json"); open(dep,"w").write(json.dumps(manifest))
        svcj= os.path.join(d,"svc.json"); open(svcj,"w").write(json.dumps(svc))
        subprocess.run(["kubectl","apply","-f",dep], check=True)
        subprocess.run(["kubectl","apply","-f",svcj], check=True)
    return {"ok":True,"name":name,"namespace":namespace}