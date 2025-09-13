# PATH: engine/memory/store.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, json, time, hashlib

ROOT = os.getenv("IMU_THREADS_DIR", ".imu/threads")
os.makedirs(ROOT, exist_ok=True)

def _safe_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def thread_dir(user_id: str, thread_id: Optional[str]) -> str:
    tid = (thread_id or "default").strip() or "default"
    d = os.path.join(ROOT, f"{_safe_id(user_id)}", tid)
    os.makedirs(d, exist_ok=True)
    return d

def _path(d: str, name: str) -> str:
    return os.path.join(d, name)

def load_recent(user_id: str, thread_id: Optional[str], n: int = 16) -> List[Dict[str, Any]]:
    d = thread_dir(user_id, thread_id)
    p = _path(d, "history.jsonl")
    if not os.path.exists(p): return []
    out: List[Dict[str,Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()[-max(0,n):]
    for ln in lines:
        try: out.append(json.loads(ln))
        except Exception: pass
    return out

def load_summary(user_id: str, thread_id: Optional[str]) -> Dict[str, Any]:
    d = thread_dir(user_id, thread_id)
    p = _path(d, "summary.json")
    if not os.path.exists(p): return {"summary": "", "decisions": [], "open_questions": []}
    try:
        return json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        return {"summary": "", "decisions": [], "open_questions": []}

def append_turn(user_id: str, thread_id: Optional[str], role: str, text: str, meta: Optional[Dict[str,Any]]=None) -> None:
    d = thread_dir(user_id, thread_id)
    p = _path(d, "history.jsonl")
    rec = {"ts": time.time(), "role": role, "text": text, "meta": (meta or {})}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _needs_rollup(d: str, every_n: int) -> bool:
    p = _path(d, "history.jsonl")
    if not os.path.exists(p): return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) >= every_n
    except Exception:
        return False

def _truncate_text(s: str, max_chars: int=1600) -> str:
    s = s.strip()
    return s if len(s) <= max_chars else (s[:max_chars] + " …")

def summarize_if_needed(user_id: str, thread_id: Optional[str], *, every_n: int=10) -> Optional[Dict[str, Any]]:
    """
    Roll up last N turns לתקציר ו'רשימת החלטות' כדי לשמר רצף אמיתי.
    משתמש ב-LLM אם קיים, אחרת מסכם נאיבית.
    """
    d = thread_dir(user_id, thread_id)
    if not _needs_rollup(d, every_n): return None

    hist = load_recent(user_id, thread_id, n=every_n*2)
    blob = "\n".join(f"{h.get('role')}: {h.get('text')}" for h in hist)
    summary, decisions, open_q = "", [], []

    try:
        # משתמש בממשק ה-LLM המקומי אם קיים
        from engine.llm_gateway import LLMGateway  # type: ignore
        gw = LLMGateway()
        messages = [
            {"role":"system","content":"סכם דיאלוג לכותרת/סיכום קצר, רשימת החלטות, ורשימת שאלות פתוחות. החזר JSON בשדות: summary, decisions[], open_questions[]."},
            {"role":"user","content": _truncate_text(blob, 3200)}
        ]
        schema = {
            "type":"object",
            "properties":{
                "summary":{"type":"string"},
                "decisions":{"type":"array", "items":{"type":"string"}},
                "open_questions":{"type":"array", "items":{"type":"string"}}
            },
            "required":["summary","decisions","open_questions"]
        }
        out = gw.chat(messages=messages, schema_hint=schema) or {}
        summary = (out.get("summary") or "").strip()
        decisions = [str(x) for x in (out.get("decisions") or [])]
        open_q = [str(x) for x in (out.get("open_questions") or [])]
    except Exception:
        # תקציר נאיבי אם אין/נפל LLM
        summary = _truncate_text(blob, 600)
        # החלטות/שאלות יהיו ריקות במקרה זה

    state = {"summary": summary, "decisions": decisions, "open_questions": open_q, "ts": time.time()}
    with open(_path(d, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return state
