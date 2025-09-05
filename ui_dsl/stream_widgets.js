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