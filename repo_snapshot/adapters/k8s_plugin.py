# adapters/k8s_plugin.py
from __future__ import annotations
from typing import Dict, Any
from .contracts.base import ResourceRequired, ProcessFailed, require_binary, run, sha256_file, BuildResult, ensure_dir, CAS_STORE
from .contracts.base import record_event, sha256_bytes
from engine.progress import EMITTER
from perf.measure import measure, JOB_PERF
import json, tempfile, os, time

def submit_k8s_job(name: str, image: str, command: list[str], namespace: str="default",
                   ttl_seconds_after_finished: int=600) -> Dict[str,Any]:
    EMITTER.emit("timeline", {"phase":"k8s.submit","job":name,"image":image})
    require_binary("kubectl","Install kubectl & configure KUBECONFIG","kubectl required for K8s")
    job = {
      "apiVersion":"batch/v1","kind":"Job",
      "metadata":{"name":name},
      "spec":{"ttlSecondsAfterFinished": ttl_seconds_after_finished,
              "template":{"spec":{"restartPolicy":"Never","containers":[{"name":name,"image":image,"command":command}]}}}
    }
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        json.dump(job,f); f.flush(); path=f.name
    try:
        run(["kubectl","apply","-n",namespace,"-f",path], timeout=30)
        t0=time.time()
        while time.time()-t0<1800:
            s=run(["kubectl","get","job",name,"-n",namespace,"-o","json"], timeout=30)
            j=json.loads(s); st=j.get("status",{})
            EMITTER.emit("progress", {"job":name,"active":st.get("active",0),"succeeded":st.get("succeeded",0),"failed":st.get("failed",0)})
            if st.get("succeeded",0)>=1 or st.get("failed",0)>=1: break
            time.sleep(2)
        pods = run(["kubectl","get","pods","-n",namespace,"-l",f"job-name={name}","-o","json"], timeout=30)
        pj=json.loads(pods); logs=[]
        for it in pj.get("items",[]):
            pn=it["metadata"]["name"]
            try:
                out = run(["kubectl","logs",pn,"-n",namespace], timeout=120)
            except ProcessFailed as e:
                out = e.err or e.out
            digest = sha256_bytes(out.encode("utf-8"))
            cas_path = os.path.join(".imu/cas", digest); ensure_dir(".imu/cas")
            with open(cas_path,"w",encoding="utf-8") as wf: wf.write(out)
            logs.append({"pod":pn,"sha256":digest})
            EMITTER.emit("logs", {"pod":pn,"len":len(out)})
        EMITTER.emit("timeline", {"phase":"k8s.done","job":name,"logs":logs})
        return {"status": "finished", "logs": logs}
    finally:
        try: os.remove(path)
        except: pass