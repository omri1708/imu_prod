import sys, yaml, json, copy, re
from typing import Any, Dict, Tuple, List

ALLOWED = {"kinds":{}, "resources":[], "fields":[], "environments":{}}
ENV_A=""; ENV_B=""

def load_docs(path:str)->List[Dict[str,Any]]:
    docs=[]
    with open(path,"r",encoding="utf-8") as f:
        for d in yaml.safe_load_all(f):
            if d: docs.append(d)
    return docs

def key(res:Dict[str,Any])->Tuple[str,str,str]:
    k=res.get("kind",""); m=res.get("metadata",{})
    return (k, m.get("namespace",""), m.get("name",""))

def scrub_generic(o:Any)->Any:
    if isinstance(o, dict):
        out={}
        for k,v in o.items():
            if k in ("creationTimestamp","resourceVersion","uid","managedFields"): continue
            if k=="annotations":
                v = {kk:vv for kk,vv in (v or {}).items() if kk.startswith("imu/")}
                if not v: continue
            out[k]=scrub_generic(v)
        return out
    if isinstance(o, list):
        return [scrub_generic(x) for x in o]
    return o

def scrub_kind(res:Dict[str,Any])->Dict[str,Any]:
    r=copy.deepcopy(res); k=r.get("kind",""); spec=r.get("spec",{}); r=scrub_generic(r)
    if k=="Deployment":
        spec.pop("replicas", None)
        tmpl=spec.get("template",{}).get("spec",{})
        for c in tmpl.get("containers",[]) or []:
            img=c.get("image")
            if isinstance(img,str) and ":" in img:
                c["image"]=img.split(":")[0]+":<tag>"
            c.pop("env", None); c.pop("resources", None)
    elif k=="Service":
        spec.pop("type", None)
    elif k=="Ingress":
        spec.pop("rules", None); spec.pop("tls", None); spec.pop("ingressClassName", None)
    elif k=="HorizontalPodAutoscaler":
        spec.pop("minReplicas", None); spec.pop("maxReplicas", None)
        t=spec.get("metrics",[])
        if t and isinstance(t,list) and t[0].get("resource",{}).get("target",{}):
            t[0]["resource"]["target"]["averageUtilization"]="<cpu>"
    elif k=="ServiceMonitor":
        spec.pop("endpoints", None)
    r["spec"]=spec; return r

def index(docs:List[Dict[str,Any]])->Dict[Tuple[str,str,str],Dict[str,Any]]:
    return { key(d): scrub_kind(d) for d in docs }

def _match_resource_allow(k:Tuple[str,str,str])->bool:
    kind, ns, name = k
    rules = (ALLOWED.get("resources") or []) + (ALLOWED.get("environments",{}).get(ENV_A,{}).get("resources",[]) or []) + (ALLOWED.get("environments",{}).get(ENV_B,{}).get("resources",[]) or [])
    for r in rules:
        if re.fullmatch(r.get("kind",".*"), kind) and re.fullmatch(r.get("namespace",".*"), ns) and re.fullmatch(r.get("name",".*"), name):
            return True
    return False

def _inject_kind_allows(obj:Dict[str,Any], env:str):
    kind=obj.get("kind","")
    def allow_list(src):
        return (src.get("kinds",{}).get(kind,[]) if src else [])
    allow = allow_list(ALLOWED) + allow_list(ALLOWED.get("environments",{}).get(env,{}))
    if not allow: return
    spec=obj.get("spec",{})
    for p in allow:
        parts=[x for x in p.split("/") if x]
        node=spec; parent=None; key=None
        # path relative to spec if not starting with "spec"
        if parts and parts[0]=="spec": parts=parts[1:]
        for i,part in enumerate(parts):
            parent=node; key=part
            if isinstance(node, list):
                idx=int(part); node = node[idx] if idx < len(node) else None
            else:
                node = node.get(part, None)
            if node is None: break
        if parent is not None and key is not None:
            if isinstance(parent, list):
                idx=int(key); 
                if idx < len(parent): parent.pop(idx)
            else:
                parent.pop(key, None)
    obj["spec"]=spec

def _inject_field_allows(k:Tuple[str,str,str], obj:Dict[str,Any], env:str):
    kind, ns, name = k
    rules = (ALLOWED.get("fields") or []) + (ALLOWED.get("environments",{}).get(env,{}).get("fields",[]) or [])
    for rule in rules:
        sel=rule.get("selector",{})
        if not re.fullmatch(sel.get("kind",".*"), kind): continue
        if not re.fullmatch(sel.get("namespace",".*"), ns): continue
        if not re.fullmatch(sel.get("name",".*"), name): continue
        for p in rule.get("allow",[]):
            parts=[x for x in p.split("/") if x]; node=obj
            for i,part in enumerate(parts):
                if i==len(parts)-1:
                    if isinstance(node, list):
                        idx=int(part); 
                        if idx < len(node): node.pop(idx)
                    elif isinstance(node, dict):
                        node.pop(part, None)
                else:
                    node = node[int(part)] if isinstance(node,list) else node.get(part, None)
                    if node is None: break

def main():
    global ALLOWED, ENV_A, ENV_B
    if len(sys.argv)<3:
        print("usage: diff_umbrella.py <envA.yaml> <envB.yaml> [allowed.yaml] [envA] [envB]", file=sys.stderr); sys.exit(2)
    allowed_path = sys.argv[3] if len(sys.argv)>3 else "scripts/allowed_diffs.yaml"
    ENV_A = sys.argv[4] if len(sys.argv)>4 else ""
    ENV_B = sys.argv[5] if len(sys.argv)>5 else ""
    try:
        with open(allowed_path,"r",encoding="utf-8") as f:
            ALLOWED = yaml.safe_load(f) or ALLOWED
    except Exception:
        pass

    ia=index(load_docs(sys.argv[1])); ib=index(load_docs(sys.argv[2]))
    keys=set(ia.keys())|set(ib.keys()); unexpected=[]
    for k in sorted(keys):
        if _match_resource_allow(k): continue
        va=ia.get(k); vb=ib.get(k)
        if va is None or vb is None: continue
        _inject_kind_allows(va, ENV_A); _inject_kind_allows(vb, ENV_B)
        _inject_field_allows(k, va, ENV_A); _inject_field_allows(k, vb, ENV_B)
        if json.dumps(va, sort_keys=True)!=json.dumps(vb, sort_keys=True):
            unexpected.append({"key":k})
    if unexpected:
        print("Unexpected diffs:"); 
        for d in unexpected[:100]: print(" -", d["key"])
        print(f"Total unexpected diffs: {len(unexpected)}"); sys.exit(1)
    print("OK: only allowed diffs across envs"); sys.exit(0)

if __name__=="__main__":
    main()