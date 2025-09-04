# imu_repo/tests/test_stage51_ledger_sync_hmac.py
from __future__ import annotations
import os, json
from audit.ledger import LEDGER_FILE, append, verify_chain
from audit.ledger_sync import export_snapshot, import_and_merge, _read_lines, _write_lines
from audit.signing import sign

def _append_signed(actor: str, action: str, obj: str):
    core = {"actor":actor,"action":action,"object":obj}
    sig  = sign(core)
    e = {**core, "_sig":sig}
    append(e)

def run():
    # נתחיל סביבת לדג'ר נקייה
    if os.path.exists(LEDGER_FILE):
        os.remove(LEDGER_FILE)

    _append_signed("n1","create","obj:1")
    _append_signed("n1","update","obj:1")
    s1 = export_snapshot("n1")

    # "צומת" אחר עם אירועים משלו
    # ניצור קובץ מרוחק ידני המדמה לדג'ר אחר
    remote = _read_lines(s1)
    core = {"actor":"n2","action":"create","object":"obj:2"}
    remote.append({**core, "_sig": sign(core), "_ts": remote[-1]["_ts"]+0.001, "_prev": remote[-1]["_hash"], "_hash": "tmp"})
    # להשלים hash חוקי:
    import hashlib, json as _json
    def _h(e): 
        base = {"_ts":e["_ts"], "_prev":e["_prev"], **{k:v for k,v in e.items() if not k.startswith("_")}}
        return hashlib.sha256(_json.dumps(base, sort_keys=True, ensure_ascii=False, separators=(",",":")).encode("utf-8")).hexdigest()
    remote[-1]["_hash"] = _h(remote[-1])
    rp = s1.replace("ledger_n1","ledger_n2")
    _write_lines(rp, remote)

    # מיזוג עם דרישת HMAC
    out = import_and_merge(rp, require_hmac=True)
    ok = verify_chain() and out["merged"]>=1
    print("OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(run())