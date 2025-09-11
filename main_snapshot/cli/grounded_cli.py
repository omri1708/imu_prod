# cli/grounded_cli.py — CLI ללא תלות חיצונית
# -*- coding: utf-8 -*-
import sys, json, base64
from provenance.store import CASStore
from engine.respond import GroundedResponder

def main():
    if len(sys.argv)<3:
        print("usage: grounded_cli.py <evidence_file> <text>"); sys.exit(2)
    ev_file = sys.argv[1]
    text = " ".join(sys.argv[2:])
    cas = CASStore(".imu_cas", ".imu_keys")
    with open(ev_file,"rb") as f:
        meta = cas.put_bytes(f.read(), sign=True, url=f"file://{ev_file}", trust=0.9, not_after_days=1)
    responder = GroundedResponder()
    out = responder.respond({"__claims__":[{"sha256": meta.sha256}]}, text)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()