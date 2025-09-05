# ui_dsl/stream_widgets.py
# -*- coding: utf-8 -*-
import json, html
from typing import List, Dict, Any

# unique IDs for multiple widgets on the same page
_TL_COUNTER = 0

def _next_id(prefix: str) -> str:
    global _TL_COUNTER
    _TL_COUNTER += 1
    return f"{prefix}{_TL_COUNTER}"



def render_progress_bars(bars: List[Dict[str, Any]]) -> str:
    """
    Stream-friendly progress bars widget.
    bars: [{"id":"p1","label":"Encode","value":35}, ...]
    - Live updates on WS topic /^progress\// with {topic:"progress/<id>", value:Number}.
    - Accessible (ARIA) and self-contained CSS+JS.
    """
    s = ['<div class="imu-progress-area" role="group" aria-label="progress group">']
    for b in bars:
        _id = html.escape(b["id"])  # required
        _label = html.escape(b.get("label", ""))
        _val = int(b.get("value", 0))
        s.append(f'''
<div class="imu-progress" id="{_id}" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{_val}">
  <div class="imu-progress__label">{_label}</div>
  <div class="imu-progress__outer"><div class="imu-progress__inner" style="width:{_val}%"></div></div>
</div>
''')
    s.append("""
<style>
.imu-progress-area{display:grid;gap:8px}
.imu-progress__outer{background:#eee;height:10px;border-radius:6px;overflow:hidden}
.imu-progress__inner{height:10px;transition:width .15s}
.imu-progress__label{font:12px sans-serif;margin-bottom:2px}
</style>
<script>
(function(){
  'use strict';
  function setVal(id, v){
    const el = document.getElementById(id);
    if(!el) return;
    const val = (v|0);
    const inner = el.querySelector('.imu-progress__inner');
    if(inner) inner.style.width = val + '%';
    el.setAttribute('aria-valuenow', String(val));
  }
  const ws = window.IMU_WS;
  if(ws && typeof ws.subscribe === 'function'){
    ws.subscribe(/^progress\//, function(msg){
      // msg: {topic:"progress/<id>", value:Number}
      const t = (msg && msg.topic) ? String(msg.topic) : '';
      const id = t.split('/')[1];
      if(!id) return;
      setVal(id, msg.value);
    });
  }
})();
</script>
""")
    s.append("</div>")
    return "".join(s)


def render_event_timeline(events: List[Dict[str, Any]]) -> str:
    """
    Stream-ready timeline widget with filters, client-side sort, and efficient incremental updates.
    - Initial events are provided by `events`.
    - Live updates arrive on WS topic /^timeline\// (expects {topic, event{ts,type,text}}) or EventSource fallback.
    - Virtualizes to keep last MAX_ITEMS to avoid DOM bloat.
    - Safe against HTML injection by using textContent only.
    """
    safe = json.dumps(events, ensure_ascii=False).replace("</", "<\/")

    tl_id = _next_id("imuTL_")
    type_id = _next_id("fltType_")
    text_id = _next_id("fltText_")
    sort_id = _next_id("fltSortTs_")

    html_doc = """
<div class="imu-timeline">
  <div class="imu-timeline__filters">
    <label>Type:
      <select id="__TYPE__">
        <option value="">Any</option><option>info</option><option>warn</option><option>error</option>
      </select>
    </label>
    <label>Search: <input id="__TEXT__" placeholder="contains..."/></label>
    <button id="__SORT__">Sort by time</button>
  </div>
  <ul id="__TL__" class="imu-timeline__list"></ul>
</div>
<style>
.imu-timeline__list{list-style:none;padding:0;margin:8px 0}
.imu-timeline__list li{padding:6px 8px;border-bottom:1px solid #eee;font:13px sans-serif}
.imu-tag-info{color:#0a0}
.imu-tag-warn{color:#c80}
.imu-tag-error{color:#c00}
</style>
<script>
(function(){
  const MAX_ITEMS = 500;             // keep up to N rendered items
  const TOPIC_RE = /^timeline\//;    // which topics to accept
  const data = __SAFE__;
  let cur = data.slice();
  let asc = true;

  const el = document.getElementById('__TL__');
  const fType = document.getElementById('__TYPE__');
  const fText = document.getElementById('__TEXT__');
  const bSort = document.getElementById('__SORT__');

  function makeItem(ev){
    const li = document.createElement('li');
    const tag = document.createElement('span');
    const strong = document.createElement('b');

    const cls = ev.type==='error'?'imu-tag-error':(ev.type==='warn'?'imu-tag-warn':'imu-tag-info');
    tag.className = cls;
    tag.textContent = ev.type || 'info';

    const dt = new Date(ev.ts*1000).toISOString();
    strong.textContent = dt;

    li.appendChild(tag);
    li.appendChild(document.createTextNode(' '));
    li.appendChild(strong);
    li.appendChild(document.createTextNode(' — '));
    li.appendChild(document.createTextNode(ev.text || ''));
    return li;
  }

  function render(list){
    const frag = document.createDocumentFragment();
    const limit = Math.min(list.length, MAX_ITEMS);
    const view = asc ? list.slice(-limit) : list.slice(0, limit);
    el.textContent = '';
    if (asc){
      for (let i=0;i<view.length;i++) frag.appendChild(makeItem(view[i]));
    } else {
      for (let i=view.length-1;i>=0;i--) frag.appendChild(makeItem(view[i]));
    }
    el.appendChild(frag);
  }

  function apply(){
    const t = fType ? fType.value : '';
    const s = (fText && fText.value ? fText.value : '').toLowerCase();
    cur = data.filter(ev => (!t || ev.type===t) && (!s || (ev.text||'').toLowerCase().includes(s)));
    cur.sort((a,b)=> asc ? (a.ts-b.ts) : (b.ts-a.ts));
    render(cur);
  }

  if (fType) fType.onchange = apply;
  if (fText) fText.oninput = apply;
  if (bSort) bSort.onclick = ()=>{ asc=!asc; apply(); };
  apply();

  // --- streaming & batching ---
  const ws = window.IMU_WS;
  let queue = [];
  let scheduled = false;
  function scheduleFlush(){
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(()=>{
      scheduled = false;
      if (!queue.length) return;
      // append and cap underlying data for memory safety
      for (let i=0;i<queue.length;i++) data.push(queue[i]);
      queue.length = 0;
      if (data.length > MAX_ITEMS*4) data.splice(0, data.length - MAX_ITEMS*2);
      apply();
    });
  }

  function onMsg(msg){
    // msg: {topic:"timeline/main", event:{ts,type,text}}
    if (!msg || !msg.topic || !TOPIC_RE.test(msg.topic)) return;
    const ev = msg.event;
    if (ev && typeof ev.ts === 'number'){
      queue.push(ev);
      scheduleFlush();
    }
  }

  if (ws && typeof ws.subscribe === 'function'){
    ws.subscribe(/^timeline\//, onMsg);
  } else if (window.IMU_STREAM_URL && window.EventSource){
    try{
      const es = new EventSource(window.IMU_STREAM_URL);
      es.onmessage = (e)=>{ try { onMsg(JSON.parse(e.data)); } catch(_){} };
    }catch(_){/* ignore */}
  }
})();
</script>
"""

    return (html_doc
            .replace("__SAFE__", safe)
            .replace("__TL__", tl_id)
            .replace("__TYPE__", type_id)
            .replace("__TEXT__", text_id)
            .replace("__SORT__", sort_id))
