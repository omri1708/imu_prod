#scripts/diff_umbrella.py (UPDATED)

import sys, yaml, json, copy, re
from typing import Any, Dict, Tuple, List, Optional

ALLOWED = {"kinds":{}, "resources":[], "fields":[]}

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
                v = {kk:vv for kk,vv in (v or {}).items() if kk.startswith("imu/")}  # נשאיר imu/* לצורך gating
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
    r= scrub_generic(r)
    # KIND-specific relaxations
    if k=="Deployment":
        spec.pop("replicas", None)
        tmpl = spec.get("template",{}).get("spec",{})
        for c in tmpl.get("containers",[]) or []:
            img=c.get("image")
            if isinstance(img,str) and ":" in img:
                c["image"]=img.split(":")[0]+":<tag>"
            c.pop("env", None)
            c.pop("resources", None)
    elif k=="Service":
        spec.pop("type", None)
    elif k=="Ingress":
        spec.pop("rules", None)
        spec.pop("tls", None)
        spec.pop("ingressClassName", None)
    elif k=="HorizontalPodAutoscaler":
        spec.pop("minReplicas", None)
        spec.pop("maxReplicas", None)
        t = spec.get("metrics",[])
        if t and isinstance(t,list) and t[0].get("resource",{}).get("target",{}):
            t[0]["resource"]["target"]["averageUtilization"]="<cpu>"
    elif k=="ServiceMonitor":
        spec.pop("endpoints", None)
    r["spec"]=spec
    return r

def index(docs:List[Dict[str,Any]])->Dict[Tuple[str,str,str],Dict[str,Any]]:
    out={}
    for d in docs:
        out[key(d)] = scrub_kind(d)
    return out

def _match_resource_allow(k:Tuple[str,str,str])->bool:
    kind, ns, name = k
    for r in ALLOWED.get("resources",[]):
        if re.fullmatch(r.get("kind",".*"), kind) and \
           re.fullmatch(r.get("namespace",".*"), ns) and \
           re.fullmatch(r.get("name",".*"), name):
            return True
    return False

def _inject_kind_allows(obj:Dict[str,Any]):
    kind=obj.get("kind","")
    allow = ALLOWED.get("kinds",{}).get(kind, [])
    if not allow: return
    # remove paths allowed (relative to spec)
    spec=obj.get("spec",{})
    for p in allow:
        try:
            parts=[x for x in p.split("/") if x]
            if not parts or parts[0]!="spec":  # relative under /spec by contract
                # allow relative to spec only; if came w/o /spec prefix, add it
                pass
            node=spec
            ok=True
            for i,part in enumerate(parts[1:] if parts and parts[0]=="spec" else parts):
                if isinstance(node, list):
                    idx=int(part)
                    if idx>=len(node): ok=False; break
                    if i==len(parts)-2:
                        node.pop(idx,None) if isinstance(node,dict) else node.pop(idx)
                    else:
                        node=node[idx]
                elif isinstance(node, dict):
                    if i==len(parts)-2:
                        node.pop(part, None)
                    else:
                        node=node.get(part, None)
                        if node is None: ok=False; break
                else:
                    ok=False; break
            # if ok: updated spec
        except Exception:
            continue
    obj["spec"]=spec

def _inject_field_allows(k:Tuple[str,str,str], obj:Dict[str,Any]):
    kind, ns, name = k
    for rule in ALLOWED.get("fields",[]):
        sel=rule.get("selector",{})
        if not re.fullmatch(sel.get("kind",".*"), kind): continue
        if not re.fullmatch(sel.get("namespace",".*"), ns): continue
        if not re.fullmatch(sel.get("name",".*"), name): continue
        for p in rule.get("allow",[]):
            # remove absolute JSON pointer (from root)
            try:
                parts=[x for x in p.split("/") if x]
                node=obj
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
            except Exception:
                continue

def main():
    if len(sys.argv) < 3:
        print("usage: diff_umbrella.py <envA.yaml> <envB.yaml> [allowed_diffs.yaml]", file=sys.stderr); sys.exit(2)
    allowed_path = sys.argv[3] if len(sys.argv)>3 else "scripts/allowed_diffs.yaml"
    try:
        with open(allowed_path,"r",encoding="utf-8") as f:
            global ALLOWED
            ALLOWED = yaml.safe_load(f) or ALLOWED
    except Exception:
        pass

    a=load_docs(sys.argv[1]); b=load_docs(sys.argv[2])
    ia=index(a); ib=index(b)
    keys=set(ia.keys())|set(ib.keys())
    unexpected=[]
    for k in sorted(keys):
        if _match_resource_allow(k): 
            continue
        va=ia.get(k); vb=ib.get(k)
        if va is None or vb is None:
            continue
        _inject_kind_allows(va); _inject_kind_allows(vb)
        _inject_field_allows(k, va); _inject_field_allows(k, vb)
        ja=json.dumps(va, sort_keys=True)
        jb=json.dumps(vb, sort_keys=True)
        if ja!=jb:
            unexpected.append({"key":k})
    if unexpected:
        print("Unexpected diffs:")
        for d in unexpected[:100]:
            print(" -", d["key"])
        print(f"Total unexpected diffs: {len(unexpected)}")
        sys.exit(1)
    print("OK: only allowed diffs across envs")
    sys.exit(0)

if __name__=="__main__":
    main()