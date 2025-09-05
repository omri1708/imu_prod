/* global EventSource */
(function(){
  const $ = (sel, root=document)=>root.querySelector(sel);
  const $$ = (sel, root=document)=>Array.from(root.querySelectorAll(sel));

  function sseConnect(topic, onMsg){
    const es = new EventSource(`/events?topic=${encodeURIComponent(topic)}`);
    es.addEventListener('msg', ev=>{
      try { onMsg(JSON.parse(ev.data)); } catch(e){}
    });
    es.addEventListener('hb', _=>{ /* heartbeat */ });
    es.onerror = _=>{ /* אפשר להוסיף backoff */ };
    return es;
  }

  // Progress bars
  class ProgressBar {
    constructor(el, opts){
      this.el = el;
      this.opts = Object.assign({label:"", min:0, max:100, value:0}, opts||{});
      this.render();
    }
    set(v){
      const min = this.opts.min, max = this.opts.max;
      const val = Math.max(min, Math.min(max, v));
      this.opts.value = val;
      this.fill.style.width = `${((val-min)/(max-min))*100}%`;
      this.txt.textContent = `${this.opts.label} ${Math.round(((val-min)/(max-min))*100)}%`;
    }
    render(){
      this.el.classList.add('imu-progress');
      this.el.innerHTML = `
        <div class="imu-progress__track">
          <div class="imu-progress__fill"></div>
        </div>
        <div class="imu-progress__text"></div>`;
      this.fill = $('.imu-progress__fill', this.el);
      this.txt  = $('.imu-progress__text', this.el);
      this.set(this.opts.value);
    }
  }

  // Event timeline (append-only)
  class EventTimeline {
    constructor(el, opts){
      this.el = el;
      this.opts = Object.assign({title:"Events", maxItems:2000}, opts||{});
      this.render();
    }
    push(ev){
      const li = document.createElement('li');
      const ts = new Date(ev.ts || Date.now()).toLocaleTimeString();
      li.innerHTML = `<b>[${ts}]</b> ${ev.kind||'event'} – ${ev.msg||''}`;
      this.list.prepend(li);
      while (this.list.children.length > this.opts.maxItems){
        this.list.removeChild(this.list.lastChild);
      }
    }
    render(){
      this.el.classList.add('imu-timeline');
      this.el.innerHTML = `<div class="imu-timeline__title">${this.opts.title}</div><ol class="imu-timeline__list"></ol>`;
      this.list = $('.imu-timeline__list', this.el);
    }
  }

  // Simple layout grid (areas + nested support via data-area)
  function applyGrid(root){
    const grids = $$('.imu-grid[data-areas]', root);
    grids.forEach(g=>{
      const areas = JSON.parse(g.getAttribute('data-areas') || '[]');
      const cols  = g.getAttribute('data-cols') || '1fr 1fr';
      const rows  = g.getAttribute('data-rows') || 'auto auto';
      g.style.display = 'grid';
      g.style.gridTemplateColumns = cols;
      g.style.gridTemplateRows = rows;
      g.style.gridTemplateAreas = areas.map(r=>`"${r.join(' ')}"`).join(' ');
      $$('.imu-cell', g).forEach(cell=>{
        const name = cell.getAttribute('data-area');
        if (name) cell.style.gridArea = name;
      });
    });
  }

  // Bootstrap page
  function boot(){
    // Progress
    const pSel = $$('.imu-progress[data-topic]');
    pSel.forEach(el=>{
      const bar = new ProgressBar(el, {label: el.getAttribute('data-label') || ''});
      sseConnect(el.getAttribute('data-topic'), msg=>{
        if (typeof msg.value === 'number') bar.set(msg.value);
      });
    });
    // Timeline
    const tSel = $$('.imu-timeline[data-topic]');
    tSel.forEach(el=>{
      const tl = new EventTimeline(el, {title: el.getAttribute('data-title') || 'Events'});
      sseConnect(el.getAttribute('data-topic'), msg=>{
        tl.push(msg);
      });
    });

    applyGrid(document);
  }

  document.addEventListener('DOMContentLoaded', boot);
})();