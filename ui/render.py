# imu_repo/ui/render.py
from __future__ import annotations
import html, json
from typing import List, Dict, Any
from grounded.claims import current
from ui.dsl import Page, Component, validate_page
from ui.forms import FormSchema, compile_schema_to_js

CSP = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'nonce-IMU_NONCE'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'; connect-src 'self'"
# Permissions-Policy בהעדפה לכותרת HTTP; כאן meta לצרכי סטטי
def _perm_policy(permissions: Dict[str, bool]) -> str:
    def v(flag: bool): return "self" if flag else "()"
    return f"geolocation=({v(permissions.get('geolocation', False))}), microphone=({v(permissions.get('microphone', False))}), camera=({v(permissions.get('camera', False))})"

def _esc(s: str) -> str:
    return html.escape(s, quote=True)

def _attrs(props: Dict[str,Any], include: List[str]) -> str:
    out = []
    for k in include:
        if k in props:
            out.append(f'{k}="{_esc(str(props[k]))}"')
    return " ".join(out)

def _render_markdown(md: str) -> str:
    # מרנדר Markdown בסיסי (כותרות/פסקאות/קישורים) ללא JS
    s = md.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    lines = s.splitlines()
    out = []
    for ln in lines:
        if ln.startswith("### "): out.append(f"<h3>{_esc(ln[4:])}</h3>")
        elif ln.startswith("## "): out.append(f"<h2>{_esc(ln[3:])}</h2>")
        elif ln.startswith("# "): out.append(f"<h1>{_esc(ln[2:])}</h1>")
        else:
            # קישורים [txt](url) — רק data:/self
            ln2 = ln
            # (פשטות: לא נכניס hrefs כאן; CSP מונע anyway)
            out.append(f"<p>{ln2}</p>")
    return "\n".join(out)

def _render_component(c: Component, out: List[str], scripts: List[str], forms_js: List[str]):
    k = c.kind
    if k == "text":
        t = _esc(c.props.get("text",""))
        out.append(f'<p id="{_esc(c.id)}" class="imu-text">{t}</p>')
    elif k == "input":
        itype = _esc(c.props.get("type","text"))
        ph = _esc(c.props.get("placeholder",""))
        name = _esc(c.props.get("name", c.id))
        out.append(f'<input id="{_esc(c.id)}" name="{name}" type="{itype}" placeholder="{ph}" class="imu-input" />')
    elif k == "button":
        label = _esc(c.props.get("label",""))
        action = _esc(c.props.get("action",""))
        out.append(f'<button id="{_esc(c.id)}" class="imu-btn" data-action="{action}">{label}</button>')
    elif k == "list":
        items = c.props.get("items",[])
        out.append(f'<ul id="{_esc(c.id)}" class="imu-list">')
        for it in items: out.append(f'  <li class="imu-li">{_esc(it)}</li>')
        out.append('</ul>')
    elif k == "image":
        src = c.props.get("src","")
        alt = _esc(c.props.get("alt",""))
        out.append(f'<img id="{_esc(c.id)}" class="imu-img" alt="{alt}" src="{_esc(src)}" />')
    elif k == "spacer":
        h = int(c.props.get("h", 12))
        out.append(f'<div id="{_esc(c.id)}" class="imu-spacer" style="height:{h}px"></div>')
    elif k == "container":
        out.append(f'<div id="{_esc(c.id)}" class="imu-container">')
        for ch in c.children: _render_component(ch, out, scripts, forms_js)
        out.append('</div>')
    elif k == "markdown":
        out.append(f'<section id="{_esc(c.id)}" class="imu-md">{_render_markdown(c.props.get("md",""))}</section>')
    elif k == "select":
        name = _esc(c.props.get("name", c.id))
        out.append(f'<select id="{_esc(c.id)}" name="{name}" class="imu-select">')
        for o in c.props.get("options", []):
            if isinstance(o, str):
                out.append(f'<option value="{_esc(o)}">{_esc(o)}</option>')
            else:
                val = _esc(str(o.get("value","")))
                lab = _esc(str(o.get("label", val)))
                out.append(f'<option value="{val}">{lab}</option>')
        out.append('</select>')
    elif k == "checkbox":
        name = _esc(c.props.get("name", c.id))
        lbl = _esc(c.props.get("label",""))
        out.append(f'<label class="imu-check"><input id="{_esc(c.id)}" name="{name}" type="checkbox" /> {lbl}</label>')
    elif k == "radio":
        name = _esc(c.props.get("name"))
        lbl = _esc(c.props.get("label",""))
        val = _esc(c.props.get("value", c.id))
        out.append(f'<label class="imu-radio"><input id="{_esc(c.id)}" name="{name}" type="radio" value="{val}" /> {lbl}</label>')
    elif k == "table":
        cols = c.props.get("columns",[])
        rows = c.props.get("rows",[])
        out.append(f'<table id="{_esc(c.id)}" class="imu-table"><thead><tr>')
        for col in cols: out.append(f'<th>{_esc(col)}</th>')
        out.append('</tr></thead><tbody>')
        for r in rows:
            out.append('<tr>')
            for cell in r: out.append(f'<td>{_esc(str(cell))}</td>')
            out.append('</tr>')
        out.append('</tbody></table>')
    elif k == "form":
        # סכמת טופס
        schema = FormSchema.from_dict(c.props.get("schema", {}))
        js_fn = compile_schema_to_js(schema)
        forms_js.append(js_fn)  # ייאוחד לבאנדל יחיד
        submit_label = _esc(c.props.get("submit_label","Submit"))
        out.append(f'<form id="{_esc(c.id)}" class="imu-form" data-imu-form="1">')
        for ch in c.children: _render_component(ch, out, scripts, forms_js)
        out.append(f'<div class="imu-form-actions"><button type="submit" class="imu-btn">{submit_label}</button></div>')
        out.append('</form>')
    else:
        # לא יגיע בגלל validate_page
        pass

def _base_css() -> str:
    return """
.imu-root{max-width:960px;margin:0 auto;padding:16px;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial,'Noto Sans',sans-serif;color:#222;background:#fff}
.imu-text{margin:8px 0;line-height:1.5}
.imu-input,.imu-select{width:100%;padding:8px;border:1px solid #ccc;border-radius:6px;margin:6px 0}
.imu-btn{padding:8px 12px;border-radius:6px;border:1px solid #1a73e8;background:#1a73e8;color:#fff;cursor:pointer}
.imu-btn:hover{filter:brightness(0.95)}
.imu-list{padding-left:18px}.imu-li{margin:4px 0}
.imu-img{max-width:100%;height:auto;border-radius:6px}
.imu-spacer{width:100%}.imu-container{display:block}
.imu-md h1,.imu-md h2,.imu-md h3{margin:12px 0 6px}
.imu-form{margin:12px 0}.imu-form-actions{margin-top:12px}
.imu-check,.imu-radio{display:block;margin:6px 0}
.imu-table{border-collapse:collapse;width:100%;margin:8px 0}
.imu-table th,.imu-table td{border:1px solid #ddd;padding:6px 8px}
.imu-error{color:#b00020;font-size:0.9em;margin:4px 0}
"""

def _base_js() -> str:
    # אירועים/אימות/הרשאות דפדפן; ללא רשת/ספריות חיצוניות.
    return r"""
(function(){
  // Evidences in-page
  window.IMU_EVIDENCES = window.IMU_EVIDENCES || [];
  function ev(kind, payload, trust){
    try{ window.IMU_EVIDENCES.push({kind, payload, trust:(trust||0.9), ts: Date.now()}); }catch(e){}
  }

  // Actions על כפתורים
  function fireAction(id, action){
    ev('ui_action', {id, action}, 0.92);
    const ce = new CustomEvent('imu:action', {detail:{id, action}});
    window.dispatchEvent(ce);
    // הרצה מובנית: הרשאות וחיישנים
    if(action === 'perm:geolocation'){ requestGeo(); }
    if(action === 'sensor:geo'){ readGeo(); }
    if(action === 'perm:camera'){ requestMedia({video:true}); }
    if(action === 'perm:microphone'){ requestMedia({audio:true}); }
  }

  document.addEventListener('click', function(evnt){
    const btn = evnt.target.closest('button[data-action]');
    if(btn){
      evnt.preventDefault();
      fireAction(btn.id, btn.getAttribute('data-action') || '');
    }
  }, true);

  // קריאת טפסים עם מאמת סכמתי
  function hydrateForms(){
    const forms = document.querySelectorAll('form[data-imu-form="1"]');
    const validators = window.__IMU_FORM_VALIDATORS__ || [];
    forms.forEach(function(f, idx){
      const validate = validators[idx] || (d => ({ok:true, errors:[]}));
      f.addEventListener('submit', function(e){
        e.preventDefault();
        // איסוף ערכים
        const fd = {};
        f.querySelectorAll('input,select,textarea').forEach(function(el){
          if(el.type === 'checkbox') fd[el.name||el.id] = !!el.checked;
          else if(el.type === 'radio'){
            if(el.checked) fd[el.name||el.id] = el.value;
          } else {
            fd[el.name||el.id] = el.value;
          }
        });
        const res = validate(fd);
        // נקה הודעות קודמות
        f.querySelectorAll('.imu-error').forEach(n => n.remove());
        if(!res.ok){
          res.errors.forEach(function(er){
            const anchor = f.querySelector('[name="'+er.field+'"],#'+er.field);
            const msg = document.createElement('div'); msg.className='imu-error';
            msg.textContent = (er.msg || 'invalid');
            if(anchor && anchor.parentNode) anchor.parentNode.appendChild(msg);
          });
          ev('ui_form_reject', {form: f.id, errors: res.errors}, 0.7);
          return;
        }
        ev('ui_form_ok', {form: f.id, data: fd}, 0.96);
        const evt = new CustomEvent('imu:form', {detail:{id: f.id, data: fd}});
        window.dispatchEvent(evt);
      }, true);
    });
  }

  // הרשאות/חיישנים — יעבדו בדפדפן אמיתי בלבד
  async function requestGeo(){
    try{
      // רק בקשת הרשאה; הדפדפן יציג prompt
      if (!navigator.permissions || !navigator.permissions.query){ return; }
      const st = await navigator.permissions.query({name:'geolocation'});
      ev('perm_status', {perm:'geolocation', state: st.state}, 0.9);
    }catch(e){ ev('perm_error', {perm:'geolocation', error: String(e)}, 0.4); }
  }
  function readGeo(){
    if (!navigator.geolocation){ ev('sensor_absent', {sensor:'geolocation'}, 0.4); return; }
    navigator.geolocation.getCurrentPosition(function(pos){
      ev('sensor_geo', {lat: pos.coords.latitude, lon: pos.coords.longitude, acc: pos.coords.accuracy}, 0.95);
      const evt = new CustomEvent('imu:sensor', {detail:{kind:'geolocation', value:{lat:pos.coords.latitude, lon:pos.coords.longitude, acc:pos.coords.accuracy}}});
      window.dispatchEvent(evt);
    }, function(err){
      ev('sensor_error', {sensor:'geolocation', error: err && err.message || String(err)}, 0.4);
    }, {enableHighAccuracy:true, maximumAge: 10000, timeout: 10000});
  }
  async function requestMedia(constraints){
    try{
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
        ev('perm_error', {perm:'media', error:'mediaDevices.getUserMedia missing'}, 0.4); return;
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      // לא מצרפים את ה-Stream לשום אלמנט כדי לשמור CSP; רק Evidence
      ev('perm_media_ok', {constraints: constraints}, 0.9);
      if (stream) stream.getTracks().forEach(t => t.stop());
    }catch(e){ ev('perm_media_error', {constraints, error:String(e)}, 0.4); }
  }

  // חשיפה גלובלית מוגבלת
  window.IMU = { fireAction, requestGeo, readGeo, requestMedia };

  // hydrate forms אחרי טעינה
  document.addEventListener('DOMContentLoaded', hydrateForms, false);
})();
"""

def render_html(page: Page, *, nonce: str="IMU_NONCE") -> str:
    validate_page(page)

    # Evidence: מבנה הדף
    comp_count = 0
    def count(c: Component):
        nonlocal comp_count
        comp_count += 1
        for ch in c.children: count(ch)
    for c in page.components: count(c)
    current().add_evidence("ui_render", {
        "source_url":"imu://ui/sandbox","trust":0.95,"ttl_s":600,
        "payload":{"title": page.title, "components": comp_count}
    })

    head = [
        '<!DOCTYPE html>','<html lang="en">','<head>',
        '  <meta charset="utf-8" />',
        f'  <meta http-equiv="Content-Security-Policy" content="{_esc(CSP)}" />',
        f'  <meta http-equiv="Permissions-Policy" content="{_esc(_perm_policy(page.permissions))}" />',
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />',
        f'  <title>{_esc(page.title)}</title>',
        f'  <style>{_base_css()}</style>',
        '</head>','<body>','<main class="imu-root">'
    ]
    body: List[str] = []
    scripts: List[str] = []
    forms_js: List[str] = []

    for comp in page.components:
        _render_component(comp, body, scripts, forms_js)

    # רישום מאמתי טפסים כרשימה גלובלית שמורה
    forms_bundle = "\n".join([f"( {js_fn} )" for js_fn in forms_js])
    tail = [
        '</main>',
        f'<script nonce="{_esc(nonce)}">{_base_js()}</script>',
        f'<script nonce="{_esc(nonce)}">window.__IMU_FORM_VALIDATORS__ = [\n{forms_bundle}\n];</script>',
        '</body>','</html>'
    ]
    return "\n".join(head + body + tail)