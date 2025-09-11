/* eslint-disable */
(function(){
  const sse = (topic, onMsg) => {
    const url = `/events?topic=${encodeURIComponent(topic)}`;
    const es = new EventSource(url);
    es.addEventListener('msg', ev => {
      try {
        const data = JSON.parse(ev.data);
        onMsg && onMsg(data);
      } catch(e){}
    });
    es.onerror = () => {}; // keep-alive by server
    return es;
  };

  class StreamProgress extends HTMLElement {
    connectedCallback(){
      this.attachShadow({mode:'open'});
      this.shadowRoot.innerHTML = `
        <style>
          .bar { height: 8px; background:#eee; border-radius:4px; }
          .fill { height: 8px; background:#4b7bec; width:0%; transition: width .2s; border-radius:4px; }
          .txt { font:12px system-ui, sans-serif; color:#333; margin-top:4px; }
        </style>
        <div class="bar"><div class="fill"></div></div>
        <div class="txt">waiting…</div>
      `;
      const fill = this.shadowRoot.querySelector('.fill');
      const txt  = this.shadowRoot.querySelector('.txt');
      const topic = this.getAttribute('topic') || 'progress';
      let cur=0, total=1;
      this._es = sse(topic, (m)=>{
        if(m.stage==='adapter_start'){ cur = (m.i||1)-1; total = m.n||1; }
        if(m.stage==='adapter_done'){ cur = m.i||1; total = m.n||1; }
        if(m.stage==='complete'){ cur = total; }
        const pct = Math.max(0, Math.min(100, Math.floor(100*cur/Math.max(1,total))));
        fill.style.width = pct + '%';
        txt.textContent = `${pct}% – ${m.stage||''} ${(m.kind||'')}`;
      });
    }
    disconnectedCallback(){
      if(this._es) this._es.close();
    }
  }
  customElements.define('stream-progress', StreamProgress);

  class EventTimeline extends HTMLElement {
    connectedCallback(){
      this.attachShadow({mode:'open'});
      this.shadowRoot.innerHTML = `
        <style>
          ul { list-style:none; padding:0; margin:0; font:12px system-ui, sans-serif; }
          li { padding:6px 8px; border-bottom:1px solid #eee; }
          .kind { color:#999; margin-left:6px; }
        </style>
        <ul></ul>
      `;
      const ul = this.shadowRoot.querySelector('ul');
      const topic = this.getAttribute('topic') || 'timeline';
      this._es = sse(topic, (m)=>{
        const li = document.createElement('li');
        const t = m.t || m.stage || 'evt';
        let line = `[${new Date().toLocaleTimeString()}] ${t}`;
        if(m.kind) line += ` · ${m.kind}`;
        if(m.err)  line += ` · ❌ ${m.err}`;
        li.textContent = line;
        ul.prepend(li);
        while(ul.children.length > 200) ul.removeChild(ul.lastChild);
      });
    }
    disconnectedCallback(){ if(this._es) this._es.close(); }
  }
  customElements.define('event-timeline', EventTimeline);
})();

// Lightweight client runtime (vanilla JS)
export function mountProgressBar(el, topic, broker) {
  let val = 0;
  const bar = document.createElement('div');
  bar.style.height = '8px';
  bar.style.background = '#eee';
  const fill = document.createElement('div');
  fill.style.height = '8px';
  fill.style.width = '0%';
  fill.style.background = '#4caf50';
  bar.appendChild(fill);
  el.appendChild(bar);
  broker.subscribe(topic, (evt) => {
    if (typeof evt.progress === 'number') {
      val = Math.max(0, Math.min(100, evt.progress));
      fill.style.width = val + '%';
    }
  });
}

export function mountEventTimeline(el, topic, broker) {
  const list = document.createElement('ul');
  list.style.listStyle = 'none';
  list.style.padding = '0';
  el.appendChild(list);
  broker.subscribe(topic, (evt) => {
    const li = document.createElement('li');
    li.textContent = `[${new Date().toISOString()}] ${evt.message || JSON.stringify(evt)}`;
    list.appendChild(li);
    if (list.childNodes.length > 200) list.removeChild(list.firstChild);
  });
}

export function mountDataTable(el, topic, broker, { freezeLeft=1 } = {}) {
  const table = document.createElement('table');
  table.style.borderCollapse = 'collapse';
  table.style.width = '100%';
  const thead = document.createElement('thead');
  const tbody = document.createElement('tbody');
  table.appendChild(thead); table.appendChild(tbody);
  el.appendChild(table);
  let columns = null;
  broker.subscribe(topic, (evt) => {
    if (evt.columns && !columns) {
      columns = evt.columns;
      const tr = document.createElement('tr');
      columns.forEach((c, idx) => {
        const th = document.createElement('th');
        th.textContent = c;
        th.style.position = idx < freezeLeft ? 'sticky' : 'static';
        th.style.left = idx < freezeLeft ? (idx*120)+'px' : '0';
        th.style.background = '#fff';
        th.style.borderBottom = '1px solid #ddd';
        th.style.padding = '4px 8px';
        tr.appendChild(th);
      });
      thead.appendChild(tr);
    }
    if (evt.row) {
      const tr = document.createElement('tr');
      evt.row.forEach((v, idx) => {
        const td = document.createElement('td');
        td.textContent = String(v);
        td.style.position = idx < freezeLeft ? 'sticky' : 'static';
        td.style.left = idx < freezeLeft ? (idx*120)+'px' : '0';
        td.style.background = idx < freezeLeft ? '#fafafa':'#fff';
        td.style.borderBottom = '1px solid #f0f0f0';
        td.style.padding = '4px 8px';
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
      if (tbody.childNodes.length > 1000) tbody.removeChild(tbody.firstChild);
    }
  });
}