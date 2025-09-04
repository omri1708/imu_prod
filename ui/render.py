# imu_repo/ui/render.py
from __future__ import annotations
import html, json, hashlib
from typing import List, Dict, Any, Tuple
from grounded.claims import current
from ui.dsl import Page, Component, validate_page

CSP = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'nonce-IMU_NONCE'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'; connect-src 'self'"

def _perm_policy(permissions: Dict[str, bool]) -> str:
    def v(flag: bool): return "self" if flag else "()"
    return f"geolocation=({v(permissions.get('geolocation', False))}), microphone=({v(permissions.get('microphone', False))}), camera=({v(permissions.get('camera', False))})"

def _esc(s: str) -> str:
    return html.escape(s, quote=True)

def _base_css() -> str:
    return """
:root{--imu-gap:12px}
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
.imu-table th{user-select:none;cursor:pointer}
.imu-table-controls{display:flex;gap:8px;align-items:center;margin:6px 0}
.imu-badge{display:inline-block;padding:2px 6px;border-radius:10px;background:#eee;font-size:12px}
.imu-grid{display:grid;grid-template-columns:repeat(var(--imu-cols,12),1fr);gap:var(--imu-gap,12px)}
"""

def _forms_runtime_js() -> str:
    # – מאמתי טפסים/הרשאות/חיישנים נשמרו כמו בשלב הקודם –
    # + פונקציות מיון/סינון לטבלאות ו-Evidences לאינטראקציות
    return r"""
(function(){
  window.IMU_EVIDENCES = window.IMU_EVIDENCES || [];
  function ev(kind, payload, trust){ try{ IMU_EVIDENCES.push({kind, payload, trust:(trust||0.9), ts: Date.now()}); }catch(e){} }

  function fireAction(id, action){
    ev('ui_action', {id, action}, 0.92);
    const ce = new CustomEvent('imu:action', {detail:{id, action}});
    window.dispatchEvent(ce);
    if(action === 'perm:geolocation'){ requestGeo(); }
    if(action === 'sensor:geo'){ readGeo(); }
    if(action === 'perm:camera'){ requestMedia({video:true}); }
    if(action === 'perm:microphone'){ requestMedia({audio:true}); }
  }

  document.addEventListener('click', function(evnt){
    const btn = evnt.target.closest('button[data-action]');
    if(btn){ evnt.preventDefault(); fireAction(btn.id, btn.getAttribute('data-action')||''); }
  }, true);

  function hydrateForms(){
    const forms = document.querySelectorAll('form[data-imu-form="1"]');
    const validators = window.__IMU_FORM_VALIDATORS__ || [];
    forms.forEach(function(f, idx){
      const validate = validators[idx] || (d => ({ok:true, errors:[]}));
      f.addEventListener('submit', function(e){
        e.preventDefault();
        const fd = {};
        f.querySelectorAll('input,select,textarea').forEach(function(el){
          if(el.type === 'checkbox') fd[el.name||el.id] = !!el.checked;
          else if(el.type === 'radio'){ if(el.checked) fd[el.name||el.id] = el.value; }
          else { fd[el.name||el.id] = el.value; }
        });
        const res = validate(fd);
        f.querySelectorAll('.imu-error').forEach(n => n.remove());
        if(!res.ok){
          res.errors.forEach(function(er){
            const anchor = f.querySelector('[name="'+er.field+'"],#'+er.field);
            const msg = document.createElement('div'); msg.className='imu-error'; msg.textContent = (er.msg||'invalid');
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

  async function requestGeo(){
    try{
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
      ev('perm_media_ok', {constraints: constraints}, 0.9);
      if (stream) stream.getTracks().forEach(t => t.stop());
    }catch(e){ ev('perm_media_error', {constraints, error:String(e)}, 0.4); }
  }

  // ===== Table sort & filter =====
  function enhanceTables(){
    document.querySelectorAll('table[data-imu-table]').forEach(function(tbl){
      const meta = JSON.parse(tbl.getAttribute('data-imu-table')||'{}');
      const thead = tbl.tHead;
      const tbody = tbl.tBodies[0];
      if(!thead || !tbody) return;
      const ncols = thead.rows[0].cells.length;
      // Global filter UI
      if(meta.filter){
        const host = document.createElement('div'); host.className='imu-table-controls';
        const inp = document.createElement('input'); inp.type='search'; inp.placeholder='Filter...'; inp.className='imu-input';
        const badge = document.createElement('span'); badge.className='imu-badge'; badge.textContent='0 matches';
        host.appendChild(inp); host.appendChild(badge);
        tbl.parentNode.insertBefore(host, tbl);
        inp.addEventListener('input', function(){
          const q = inp.value.trim().toLowerCase();
          let shown = 0;
          [...tbody.rows].forEach(function(r){
            const txt = [...r.cells].map(td => td.textContent.toLowerCase()).join(' ');
            const ok = !q || txt.indexOf(q) >= 0;
            r.style.display = ok? '' : 'none';
            if(ok) shown++;
          });
          badge.textContent = shown + ' matches';
          ev('ui_table_filter', {id: tbl.id, q, shown}, 0.88);
        });
      }
      // Sort on header click
      if(meta.sortable){
        [...thead.rows[0].cells].forEach(function(th, idx){
          let dir = 0; // 0 none, 1 asc, -1 desc
          th.addEventListener('click', function(){
            dir = (dir===1 ? -1 : 1);
            const rows = [...tbody.rows];
            rows.sort(function(a,b){
              const av = (a.cells[idx]?.textContent||'').trim();
              const bv = (b.cells[idx]?.textContent||'').trim();
              const na = parseFloat(av), nb = parseFloat(bv);
              const bothNum = isFinite(na) && isFinite(nb);
              const cmp = bothNum ? (na-nb) : av.localeCompare(bv, undefined, {numeric:true,sensitivity:'base'});
              return dir * (cmp<0? -1 : cmp>0? 1 : 0);
            });
            rows.forEach(r => tbody.appendChild(r));
            ev('ui_table_sort', {id: tbl.id, col: idx, dir}, 0.9);
          });
        });
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function(){
    hydrateForms();
    enhanceTables();
  }, false);

  // export
  window.IMU = { fireAction, requestGeo, readGeo, requestMedia };
})();
"""

def _render_markdown(md: str) -> str:
    s = md.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    out = []
    for ln in s.splitlines():
        if ln.startswith("### "): out.append(f"<h3>{_esc(ln[4:])}</h3>")
        elif ln.startswith("## "): out.append(f"<h2>{_esc(ln[3:])}</h2>")
        elif ln.startswith("# "): out.append(f"<h1>{_esc(ln[2:])}</h1>")
        else: out.append(f"<p>{ln}</p>")
    return "\n".join(out)

def _collect_grid_css(page: Page) -> str:
    """
    יוצר CSS פר-קומפוננטה (ids) כדי לתמוך ב-span רספונסיבי לכל col.
    """
    rules: List[str] = []
    # ברייקפוינטים ברירת מחדל:
    def_bp = {"sm":480,"md":768,"lg":1024,"xl":1440}

    def walk(c: Component, grid_ctx: Dict[str,Any]|None):
        if c.kind == "grid":
            cols = int(c.props.get("cols",12))
            gap = int(c.props.get("gap",12))
            bps = c.props.get("breakpoints", def_bp)
            rules.append(f"#{_esc(c.id)}"+"{--imu-cols:"+str(cols)+";--imu-gap:"+str(gap)+"px}")
            # המשך עם ההקשר
            ctx = {"bps": bps}
            for ch in c.children: walk(ch, ctx)
            return
        if c.kind == "col":
            span = c.props.get("span", 12)
            if isinstance(span, dict):
                # בסיס: xs
                xs = span.get("xs", span.get("sm", span.get("md", 12)))
                rules.append(f"#{_esc(c.id)}"+"{grid-column:span "+str(int(xs))+"}")
                # מדיה לכל bp
                bps = (grid_ctx or {}).get("bps", def_bp)
                for name, px in bps.items():
                    if name in span:
                        rules.append(f"@media (min-width:{int(px)}px){{#{_esc(c.id)}"+"{grid-column:span "+str(int(span[name]))+"}}}")
            else:
                rules.append(f"#{_esc(c.id)}"+"{grid-column:span "+str(int(span))+"}")
        for ch in c.children:
            walk(ch, grid_ctx)

    for c in page.components:
        walk(c, None)
    return "\n".join(rules)

def render_html(page: Page, *, nonce: str="IMU_NONCE") -> str:
    validate_page(page)

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
        f'  <style id="imu-grid-css">{_collect_grid_css(page)}</style>',
        '</head>','<body>','<main class="imu-root">'
    ]
    body: List[str] = []
    forms_js_bundle: List[str] = []

    def render_comp(c: Component):
        k = c.kind
        if k == "text":
            body.append(f'<p id="{_esc(c.id)}" class="imu-text">{_esc(c.props.get("text",""))}</p>')
        elif k == "input":
            itype = _esc(c.props.get("type","text"))
            ph = _esc(c.props.get("placeholder",""))
            name = _esc(c.props.get("name", c.id))
            body.append(f'<input id="{_esc(c.id)}" name="{name}" type="{itype}" placeholder="{ph}" class="imu-input" />')
        elif k == "button":
            label = _esc(c.props.get("label",""))
            action = _esc(c.props.get("action",""))
            body.append(f'<button id="{_esc(c.id)}" class="imu-btn" data-action="{action}">{label}</button>')
        elif k == "list":
            items = c.props.get("items",[])
            body.append(f'<ul id="{_esc(c.id)}" class="imu-list">')
            for it in items: body.append(f'  <li class="imu-li">{_esc(it)}</li>')
            body.append('</ul>')
        elif k == "image":
            src = c.props.get("src","")
            alt = _esc(c.props.get("alt",""))
            body.append(f'<img id="{_esc(c.id)}" class="imu-img" alt="{alt}" src="{_esc(src)}" />')
        elif k == "spacer":
            h = int(c.props.get("h", 12))
            body.append(f'<div id="{_esc(c.id)}" class="imu-spacer" style="height:{h}px"></div>')
        elif k == "container":
            body.append(f'<div id="{_esc(c.id)}" class="imu-container">')
            for ch in c.children: render_comp(ch)
            body.append('</div>')
        elif k == "markdown":
            body.append(f'<section id="{_esc(c.id)}" class="imu-md">{_esc(_render_markdown(c.props.get("md","")))}</section>')
        elif k == "select":
            name = _esc(c.props.get("name", c.id))
            body.append(f'<select id="{_esc(c.id)}" name="{name}" class="imu-select">')
            for o in c.props.get("options", []):
                if isinstance(o, str):
                    body.append(f'<option value="{_esc(o)}">{_esc(o)}</option>')
                else:
                    val = _esc(str(o.get("value",""))); lab = _esc(str(o.get("label", val)))
                    body.append(f'<option value="{val}">{lab}</option>')
            body.append('</select>')
        elif k == "checkbox":
            name = _esc(c.props.get("name", c.id)); lbl = _esc(c.props.get("label",""))
            body.append(f'<label class="imu-check"><input id="{_esc(c.id)}" name="{name}" type="checkbox" /> {lbl}</label>')
        elif k == "radio":
            name = _esc(c.props.get("name")); lbl = _esc(c.props.get("label","")); val = _esc(c.props.get("value", c.id))
            body.append(f'<label class="imu-radio"><input id="{_esc(c.id)}" name="{name}" type="radio" value="{val}" /> {lbl}</label>')
        elif k == "table":
            cols = c.props.get("columns",[])
            rows = c.props.get("rows",[])
            meta = {"filter": bool(c.props.get("filter", False)), "sortable": bool(c.props.get("sortable", True))}
            body.append(f'<table id="{_esc(c.id)}" class="imu-table" data-imu-table="{_esc(json.dumps(meta))}"><thead><tr>')
            for col in cols: body.append(f'<th>{_esc(col)}</th>')
            body.append('</tr></thead><tbody>')
            for r in rows:
                body.append('<tr>')
                for cell in r: body.append(f'<td>{_esc(str(cell))}</td>')
                body.append('</tr>')
            body.append('</tbody></table>')
            current().add_evidence("ui_table_render", {
                "source_url":"imu://ui/table","trust":0.94,"ttl_s":600,
                "payload":{"id": c.id, "rows": len(rows), "cols": len(cols), **meta}
            })
        elif k == "form":
            from ui.forms import FormSchema, compile_schema_to_js
            schema = FormSchema.from_dict(c.props.get("schema", {}))
            js_fn = compile_schema_to_js(schema)
            forms_js_bundle.append(js_fn)
            body.append(f'<form id="{_esc(c.id)}" class="imu-form" data-imu-form="1">')
            for ch in c.children: render_comp(ch)
            submit_label = _esc(c.props.get("submit_label","Submit"))
            body.append(f'<div class="imu-form-actions"><button type="submit" class="imu-btn">{submit_label}</button></div>')
            body.append('</form>')
        elif k == "grid":
            body.append(f'<div id="{_esc(c.id)}" class="imu-grid">')
            for ch in c.children: render_comp(ch)
            body.append('</div>')
        elif k == "col":
            body.append(f'<div id="{_esc(c.id)}">')
            for ch in c.children: render_comp(ch)
            body.append('</div>')
        else:
            pass

    for comp in page.components:
        render_comp(comp)

    forms_bundle = "\n".join([f"( {js_fn} )" for js_fn in forms_js_bundle])
    tail = [
        '</main>',
        f'<script nonce="{_esc(nonce)}">{_forms_runtime_js()}</script>',
        f'<script nonce="{_esc(nonce)}">window.__IMU_FORM_VALIDATORS__ = [\n{forms_bundle}\n];</script>',
        '</body>','</html>'
    ]
    return "\n".join(head + body + tail)