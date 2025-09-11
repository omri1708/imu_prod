# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import difflib, re
from engine.llm_gateway import LLMGateway

def classify_failure(build: Dict[str, Any]) -> Dict[str, str]:
    out = (build.get("compile_out") or "") + "\n" + (build.get("test_out") or "")
    s = out.lower()
    if "modulenotfounderror" in s or "no module named" in s: return {"kind": "deps", "hint": "missing python module"}
    if "syntaxerror" in s or "indentationerror" in s or " error:" in s: return {"kind": "compile", "hint": "syntax"}
    if "assert" in s or "failed" in s: return {"kind": "test", "hint": "assertion"}
    if "connection refused" in s or "timeout" in s: return {"kind": "infra", "hint": "service/port not up"}
    return {"kind": "unknown", "hint": ""}

def _apply_unified_diff(files: Dict[str, bytes|str], patch_text: str) -> Optional[Dict[str, bytes]]:
    parts = re.split(r'(?m)^diff --git .*$', patch_text)
    out = {k:(v.encode("utf-8") if isinstance(v,str) else v) for k,v in files.items()}
    if not parts: return None
    ok_any = False
    for pt in parts:
        m_new = re.search(r'^\+\+\+ b/(.+)$', pt, re.M) or re.search(r'^\+\+\+ (.+)$', pt, re.M)
        if not m_new: continue
        path = m_new.group(1).strip()
        # כאן אפשר לשפר parser; לבינתיים נחיל טקסט חדש רק אם ה‑diff כולל קוֹנטקסט מלא
        body = []
        for ln in pt.splitlines(True):
            if ln.startswith('+') and not ln.startswith('+++ '): body.append(ln[1:])
            elif ln.startswith(' ') or ln.startswith('@@'): body.append(ln[1:] if ln.startswith(' ') else '')
        if not body: continue
        out[path] = "".join(body).encode("utf-8")
        ok_any = True
    return out if ok_any else None

def self_heal_once(spec: Dict[str, Any], files: Dict[str, bytes|str], build: Dict[str, Any], cls: Dict[str, str]) -> Tuple[Optional[Dict[str, bytes]], str]:
    kind = cls.get("kind","unknown")
    gw = LLMGateway()
    if kind == "deps":
        prompt = ("Missing Python module. Return a minimal unified diff that adds the package to requirements.txt "
                  "OR refactors imports to available modules. Only the diff.")
    elif kind == "compile":
        prompt = ("Fix Python SyntaxError/IndentationError via minimal unified diff. Only the diff.")
    elif kind == "test":
        prompt = ("Tests failing. Return minimal unified diff to production code (not tests) to satisfy behavior. Only the diff.")
    elif kind == "infra":
        prompt = ("Infra connectivity failure. Suggest a single-file config change (port/env) as unified diff. Only the diff.")
    else:
        return None, "unknown failure"

    content = {
        "prompt": prompt + "\n\nBuild logs:\n```\n" +
                  ((build.get("compile_out") or "") + "\n" + (build.get("test_out") or ""))[:4000] + "\n```" +
                  "\n\nFiles:\n" + "\n".join(sorted(files.keys()))[:4000]
    }
    out = gw.chat(user_id=spec.get("user_id","user"), task="fix", intent="code_diff",
                  content=content, require_grounding=False, temperature=0.0)
    if not out.get("ok"): return None, "no patch"
    txt = (out["payload"].get("text") or "").strip()
    if not txt or ("+++" not in txt or "---" not in txt):
        return None, "no patch content"
    patched = _apply_unified_diff(files, txt)
    if not patched: return None, "patch failed to apply"
    return patched, "patched"
