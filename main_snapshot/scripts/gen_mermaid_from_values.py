# scripts/gen_mermaid_from_values.py

# מחולל Mermaid מה-helm/umbrella values.*.yaml: מצייר יחסי תלות ותצורות עיקריות.
import sys, os, yaml, textwrap, argparse
from pathlib import Path

def load_values(files):
    merged={}
    for f in files:
        if not Path(f).exists(): continue
        with open(f,"r",encoding="utf-8") as h:
            y=yaml.safe_load(h) or {}
            merged = deep_merge(merged, y)
    return merged

def deep_merge(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        o=a.copy()
        for k,v in b.items():
            o[k] = deep_merge(o.get(k), v)
        return o
    return b

def mermaid(Values):
    ns = Values.get("namespace","default")
    cp  = Values.get("controlPlane",{}).get("enabled",False)
    mon = Values.get("monitoring",{}).get("enabled",False)
    gk  = Values.get("gatekeeper",{}).get("enabled",False)
    ed  = Values.get("externalDNS",{}).get("enabled",False)
    ing = Values.get("ingressNginx",{}).get("enabled",False)
    cm  = Values.get("certManager",{}).get("enabled",False)
    lok = Values.get("loki",{}).get("enabled",False)

    nodes=[]
    edges=[]
    def N(k, label):
        nid=k.replace(".","_")
        nodes.append(f'{nid}["{label}"]')
        return nid
    umb = N("umbrella","Umbrella")
    if cp: edges.append(f'{umb}-->'+N("cp","Control-Plane"))
    if mon: edges.append(f'{umb}-->'+N("mon","Monitoring (kube-prom-stack)"))
    if gk: edges.append(f'{umb}-->'+N("gk","Gatekeeper (OPA)"))
    if ed: edges.append(f'{umb}-->'+N("ed","ExternalDNS"))
    if ing: edges.append(f'{umb}-->'+N("ing","Ingress-Nginx"))
    if cm: edges.append(f'{umb}-->'+N("cm","cert-manager"))
    if lok: edges.append(f'{umb}-->'+N("loki","Loki/Promtail"))

    if cp:
        cpvals = Values.get("controlPlane",{}).get("imu-control-plane",{})
        svc = f'{cpvals.get("namespace",ns)} svc'
        nodes.append(f'api["API {svc}"]'); nodes.append(f'ws["WS WFQ"]'); nodes.append('ui["UI Static"]')
        edges += ['cp-->api','cp-->ws','cp-->ui']

    return "flowchart LR\n  " + "\n  ".join(nodes+edges) + "\n"

def write_md(out_md, diagram):
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Umbrella Diagram (auto)\n\n")
        f.write("```mermaid\n")
        f.write(diagram)
        f.write("```\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--values", nargs="+", default=["helm/umbrella/values.yaml"])
    ap.add_argument("--out", default="docs/diagrams/generated/umbrella.md")
    args=ap.parse_args()
    vals=load_values(args.values)
    diag=mermaid(vals)
    write_md(args.out, diag)
    print(f"Generated {args.out}")
if __name__=="__main__":
    main()