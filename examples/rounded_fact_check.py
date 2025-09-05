# examples/grounded_fact_check.py
# -*- coding: utf-8 -*-
import json, re
from grounded.http_verifier import http_head_exists

def answer_with_claim(text: str, url: str):
    """
    עונה רק אם קיים מקור מאומת (HEAD 200) ל-URL המסופק.
    זה usage קטן שמראה אכיפה "אפס הלוצינציות": אין URL תקף → אין תשובה.
    """
    ok = http_head_exists(url, timeout_sec=3.0)
    if not ok:
        return {"ok": False, "reason": "no_evidence", "url": url}
    # דוגמה: איסוף claim+evidence
    claim = {"text": text, "url": url}
    return {"ok": True, "answer": text, "claims": [claim]}

if __name__ == "__main__":
    r1 = answer_with_claim("The IMU site is available.", "https://example.org/")
    print(json.dumps(r1, indent=2))
    r2 = answer_with_claim("This URL does not exist.", "https://example.invalid/404")
    print(json.dumps(r2, indent=2))