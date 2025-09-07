# scripts/diff_umbrella.py
"""
Compare rendered Helm YAMLs across envs and fail on unexpected diffs.
Allowed differences:
- metadata.namespace / labels / annotations
- Deployment.spec.replicas
- Container images, env, resources
- Service.spec.type
- Ingress rules/tls/annotations/class
- HPA targets (min/max/cpu)
- ServiceMonitor scrape attrs
- ExternalDNS args domain filters/txtOwnerId/policy
"""
import sys, yaml, json, copy, re
from typing import Any, Dict, Tuple, List

def load_docs(path:str)->List[Dict[str,Any]]:
    docs=[]
    with open(path,"r",encoding="utf-8") as f:
        for d in yaml.safe_load_all(f):
            if d: docs.append(d)
    return docs

def key(res:Dict[str,Any])->Tuple[str,str,str]:
    k = res.get("kind","")
    meta = res.get("metadata",{})
    return (k, meta.get("namespace",""), meta.get("name",""))

def scrub_generic(o:Any)->Any:
    if isinstance(o, dict):
        out={}
        for k,v in o.items():
            if k in ("creationTimestamp","resourceVersion","uid","managedFields"):
                continue
            if k=="annotations":
                # keep only imu/* annotations (gating) – ignore the rest
                v = {kk:vv for kk,vv in (v or {}).items() if kk.startswith("imu/")}
                if not v: continue
            out[k]=scrub_generic(v)
        return out
    if isinstance(o, list):
        return [scrub_generic(x) for x in o]
    return o

def scrub_kind(res:Dict[str,Any])->Dict[str,Any]:
    r=copy.deepcopy(res)
    k=r.get("kind","")
    spec=r.get("spec",{})
    # universal scrubs
    r= scrub_generic(r)
    # KIND-specific relaxations
    if k=="Deployment":
        # replicas allowed to differ
        spec.pop("replicas", None)
        # containers – strip image tags, allow env/resources diffs
        tmpl = spec.get("template",{}).get("spec",{})
        for c in tmpl.get("containers",[]) or []:
            img=c.get("image")
            if isinstance(img,str) and ":" in img:
                c["image"]=img.split(":")[0]+":<tag>"
        # remove env/resources to avoid noise (optional)
        for c in tmpl.get("containers",[]) or []:
            c.pop("env", None)
            c.pop("resources", None)
    elif k=="Service":
        # allow type differences
        spec.pop("type", None)
    elif k=="Ingress":
        # allow rules/tls/className/annotations (keep only structure)
        spec.pop("rules", None)
        spec.pop("tls", None)
        spec.pop("ingressClassName", None)
    elif k=="HorizontalPodAutoscaler":
        # allow numeric HPA targets
        spec.pop("minReplicas", None)
        spec.pop("maxReplicas", None)
        t = spec.get("metrics",[])
        if t and isinstance(t,list) and t[0].get("resource",{}).get("target",{}):
            t[0]["resource"]["target"]["averageUtilization"]="<cpu>"
    elif k=="ServiceMonitor":
        # allow endpoint scrape settings
        spec.pop("endpoints", None)
    elif k=="Deployment" and r.get("metadata",{}).get("name","").startswith("external-dns"):
        # ignore args differences
        pass
    r["spec"]=spec
    return r

def index(docs:List[Dict[str,Any]])->Dict[Tuple[str,str,str],Dict[str,Any]]:
    out={}
    for d in docs:
        out[key(d)] = scrub_kind(d)
    return out

def main():
    if len(sys.argv)!=3:
        print("usage: diff_umbrella.py <envA.yaml> <envB.yaml>", file=sys.stderr); sys.exit(2)
    a=load_docs(sys.argv[1]); b=load_docs(sys.argv[2])
    ia=index(a); ib=index(b)
    keys=set(ia.keys())|set(ib.keys())
    unexpected=[]
    for k in sorted(keys):
        va=ia.get(k); vb=ib.get(k)
        if va is None or vb is None:
            # resource added/removed across envs – allowed for dev/staging vs prod
            continue
        ja=json.dumps(va, sort_keys=True)
        jb=json.dumps(vb, sort_keys=True)
        if ja!=jb:
            unexpected.append({"key":k, "diff":"DIFF"})
    if unexpected:
        print("Unexpected diffs (filtered):")
        for d in unexpected[:50]:
            print(" -", d["key"])
        print(f"Total unexpected diffs: {len(unexpected)}")
        sys.exit(1)
    print("OK: only allowed diffs across envs")
    sys.exit(0)

if __name__=="__main__":
    main()