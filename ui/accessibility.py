from __future__ import annotations
from typing import Dict, Any, List
import re, os

def _text_between(s: str, tag: str) -> str:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", s, re.I|re.S)
    return (m.group(1).strip() if m else "")

def analyze_html(html: str) -> Dict[str,Any]:
    score = 100
    violations: List[str] = []
    # lang
    if not re.search(r"<html[^>]*\blang\s*=\s*['\"][a-z-]+['\"][^>]*>", html, re.I):
        score -= 20; violations.append("missing_lang")
    # title
    if len(_text_between(html, "title")) == 0:
        score -= 15; violations.append("missing_title")
    # meta description
    if not re.search(r"<meta[^>]*name=['\"]description['\"][^>]*>", html, re.I):
        score -= 10; violations.append("missing_meta_description")
    # buttons accessible name
    if re.search(r"<button[^>]*>(\s*)</button>", html, re.I):
        score -= 10; violations.append("empty_button_text")
    # aria-live region
    if "aria-live" not in html:
        score -= 5; violations.append("missing_aria_live")
    score = max(0, min(100, score))
    return {"score": score, "violations": violations}

def analyze_ui_folder(ui_root: str) -> Dict[str,Any]:
    index = os.path.join(ui_root, "index.html")
    try:
        with open(index,"r",encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        return {"score": 0, "violations": ["missing_index_html"]}
    return analyze_html(html)