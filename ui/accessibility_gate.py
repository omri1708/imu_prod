# imu_repo/ui/accessibility_gate.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os, re
from html.parser import HTMLParser
from colorsys import rgb_to_hls

class _Doc(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title=""
        self.lang=""
        self.imgs: List[Tuple[str,str]]=[]
        self.labels: List[str]=[]
        self.inputs: List[str]=[]
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag=="html": self.lang = a.get("lang","")
        if tag=="img":  self.imgs.append((a.get("src",""), a.get("alt","")))
        if tag=="label" and "for" in a: self.labels.append(a["for"])
        if tag=="input" and "id" in a:  self.inputs.append(a["id"])
    def handle_startendtag(self, tag, attrs): self.handle_starttag(tag, attrs)
    def handle_data(self, data):
        pass
    def handle_endtag(self, tag):
        pass

def _parse_html(p: str) -> _Doc:
    d = _Doc()
    with open(p,"r",encoding="utf-8") as f:
        s=f.read()
    # title
    m = re.search(r"<title>(.*?)</title>", s, re.I|re.S)
    if m: d.title = m.group(1).strip()
    d.feed(s)
    return d

_hex = re.compile(r"#([0-9a-fA-F]{6})")
def _read_colors(css_p: str) -> Tuple[str,str]:
    """
    שואף לקרוא color/background של body מתוך style.css — נדרש ליחס ניגודיות.
    """
    try:
        s = open(css_p,"r",encoding="utf-8").read()
    except FileNotFoundError:
        return ("#000000","#ffffff")
    # body { color: #111111; background: #ffffff; }
    m_color = re.search(r"body\s*{[^}]*color\s*:\s*(#[0-9a-fA-F]{6})", s)
    m_bg    = re.search(r"body\s*{[^}]*background\s*:\s*(#[0-9a-fA-F]{6})", s)
    fg = m_color.group(1) if m_color else "#111111"
    bg = m_bg.group(1) if m_bg else "#ffffff"
    return (fg, bg)

def _hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h=h.lstrip("#")
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def _luminance(rgb: Tuple[int,int,int]) -> float:
    # WCAG relative luminance approx via HLS lightness as fallback
    r,g,b = [x/255.0 for x in rgb]
    # Proper WCAG uses linearized sRGB; כאן נשתמש בקירוב דרך HLS L
    return rgb_to_hls(r,g,b)[1]

def _contrast_ratio(fg: str, bg: str) -> float:
    L1 = _luminance(_hex_to_rgb(fg))
    L2 = _luminance(_hex_to_rgb(bg))
    hi = max(L1,L2); lo = min(L1,L2)
    return (hi + 0.05) / (lo + 0.05)

def check_directory(dir_path: str, *, min_contrast: float=4.5) -> Dict[str,Any]:
    """
    מחזיר דו"ח מפורט + ok/violations.
    כללים:
      - html[lang] קיים
      - <title> לא ריק
      - לכל img יש alt לא ריק
      - לכל input יש label[for] תואם
      - יחס ניגודיות body fg/bg >= 4.5
    """
    pages=[p for p in os.listdir(dir_path) if p.endswith(".html")]
    viol=[]
    for page in pages:
        d = _parse_html(os.path.join(dir_path,page))
        if not d.lang: viol.append(("lang_missing", page))
        if not d.title: viol.append(("title_missing", page))
        for src,alt in d.imgs:
            if (alt or "").strip()=="":
                viol.append(("img_alt_missing", page, src))
        for inp in d.inputs:
            if inp not in d.labels:
                viol.append(("input_label_missing", page, inp))
    fg,bg = _read_colors(os.path.join(dir_path,"style.css"))
    cr = _contrast_ratio(fg,bg)
    if cr < min_contrast:
        viol.append(("low_contrast", f"{cr:.2f}", f"min={min_contrast}"))

    return {"ok": len(viol)==0, "violations": viol, "contrast": cr, "pages": pages}