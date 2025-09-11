from __future__ import annotations
from adapters.fs_sandbox import write_text, read_text, delete_path, FSAccessDenied
from adapters.net_sandbox import NetSandbox, NetDenied, NetRateLimit

def run():
    # FS
    write_text("workspace/hello.txt", "hi", ttl_sec=3)
    assert read_text("workspace/hello.txt") == "hi"
    # מחיקת נתיב לא מותר
    try:
        write_text("../escape.txt","x")
        return 1
    except FSAccessDenied:
        pass

    # NET — דומיין מותר? (לפי policy: *.gov מותר)
    try:
        txt = NetSandbox.http_get_text("https://example.gov/")
        assert isinstance(txt, str)
    except NetDenied:
        # אם המדיניות שלך לא כוללת example.gov — עדכן policy
        pass
    # קצב
    ok=0; fail=0
    for i in range(50):
        try:
            NetSandbox.http_get_text("https://example.gov/")
            ok+=1
        except NetRateLimit:
            fail+=1
    print("rate:", ok, fail)
    return 0

if __name__=="__main__":
    raise SystemExit(run())