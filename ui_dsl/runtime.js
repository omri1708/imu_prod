/* global EventSource */
(function(){
  const $ = (sel, root=document)=>root.querySelector(sel);
  const $$ = (sel, root=document)=>Array.from(root.querySelectorAll(sel));

  function sseConnect(topic, onMsg){
    const es = new EventSource(`/events?topic=${encodeURIComponent(topic)}`);
    es.addEventListener('msg', ev=>{
      try { onMsg(JSON.parse(ev.data)); } catch(e){}
    });
    es.addEventListener('hb', _=>{});
    es.onerror = _=>{};
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
  ProgressBar.prototype.set = function(v){
    const min=this.opts.min,max=this.opts.max; const val=Math.max(min,Math.min(max,v));
    this.opts.value=val; this.fill.style.width=`${((val-min)/(max-min))*100}%`;
    this.txt.textContent=`${this.opts.label} ${Math.round(((val-min)/(max-min))*100)}%`;
  };
  ProgressBar.prototype.render = function(){
    this.el.classList.add('imu-progress'); this.el.innerHTML =
      `<div class="imu-progress__track"><div class="imu-progress__fill"></div></div>
       <div class="imu-progress__text"></div>`;
    this.fill=$('.imu-progress__fill',this.el); this.txt=$('.imu-progress__text',this.el);
    this.set(this.opts.value);
  };
  function Progress(el,opts){ this.el=el; this.opts=Object.assign({label:"",min:0,max:100,value:0},opts||{}); this.render();}
  Progress.prototype=ProgressBar.prototype;
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

    EventTimeline.prototype.push=function(ev){
    const li=document.createElement('li'); const ts=new Date(ev.ts||Date.now()).toLocaleTimeString();
    li.innerHTML=`<b>[${ts}]</b> ${ev.kind||'event'} – ${ev.msg||''}`; this.list.prepend(li);
    while(this.list.children.length>this.opts.maxItems){ this.list.removeChild(this.list.lastChild); }
  };
  EventTimeline.prototype.render=function(){
    this.el.classList.add('imu-timeline'); this.el.innerHTML =
      `<div class="imu-timeline__title">${this.opts.title}</div><ol class="imu-timeline__list"></ol>`;
    this.list=$('.imu-timeline__list',this.el);
  };

  class StreamingTable {
    constructor(el, opts){
      this.el=el; this.opts=Object.assign({columns:[], freeze:0, maxRows:2000}, opts||{});
      this.rows=[]; this.filters={}; this.sortKey=null; this.sortDir=1;
      this.render();
    }
    render(){
      this.el.classList.add('imu-table');
      const head = `<thead><tr>${
        this.opts.columns.map((c,i)=>`<th data-key="${c.key}" ${i<this.opts.freeze?'class="freeze"':''}>
          ${c.title||c.key}<button data-sort="${c.key}">↕</button></th>`).join('')}
      </tr><tr class="filters">${
        this.opts.columns.map((c,i)=>`<th ${i<this.opts.freeze?'class="freeze"':''}>
          <input data-filter="${c.key}" placeholder="filter ${c.title||c.key}"/></th>`).join('')}
      </tr></thead>`;
      this.el.innerHTML = `<table>${head}<tbody></tbody></table>`;
      this.tbody = $('tbody', this.el);
      // אירועים
      this.el.addEventListener('click', (e)=>{
        const key = e.target.getAttribute('data-sort'); if(!key) return;
        this.sortDir = (this.sortKey===key)? -this.sortDir : 1; this.sortKey = key; this._rerender();
      });
      this.el.addEventListener('input', (e)=>{
        const key = e.target.getAttribute('data-filter'); if(!key) return;
        this.filters[key] = (e.target.value||"").toLowerCase(); this._rerender();
      });
    }
    push(row){
      this.rows.unshift(row);
      if (this.rows.length>this.opts.maxRows) this.rows.pop();
      this._rerender();
    }
    _passFilters(row){
      for(const [k,v] of Object.entries(this.filters)){
        if(!v) continue;
        const cell = (row[k]==null?"":String(row[k])).toLowerCase();
        if(!cell.includes(v)) return false;
      }
      return true;
    }
    _rerender(){
      let data = this.rows.filter(r=>this._passFilters(r));
      if (this.sortKey){
        const k=this.sortKey, dir=this.sortDir;
        data = data.slice().sort((a,b)=>{
          const va=a[k], vb=b[k];
          if(va==vb) return 0; return (va<vb?-1:1)*dir;
        });
      }
      const cols=this.opts.columns;
      this.tbody.innerHTML = data.map(r=>{
        return `<tr>${cols.map((c,i)=>`<td ${i<this.opts.freeze?'class="freeze"':''}>
          ${r[c.key]==null?"":String(r[c.key])}</td>`).join('')}</tr>`;
      }).join('');
    }
  }

  function applyGrid(root){
    const grids = $$('.imu-grid[data-areas]', root);
    grids.forEach(g=>{
      const areas = JSON.parse(g.getAttribute('data-areas') || '[]');
      const cols  = g.getAttribute('data-cols') || '1fr';
      const rows  = g.getAttribute('data-rows') || 'auto';
      g.style.display='grid';
      g.style.gridTemplateColumns = cols;
      g.style.gridTemplateRows = rows;
      g.style.gridTemplateAreas = areas.map(r=>`"${r.join(' ')}"`).join(' ');
      $$('.imu-cell', g).forEach(cell=>{
        const name=cell.getAttribute('data-area'); if(name) cell.style.gridArea=name;
      });
    });
  }

  function boot(){
    // progress
    $$('.imu-progress[data-topic]').forEach(el=>{
      const bar = new Progress(el, {label: el.getAttribute('data-label')||''});
      sseConnect(el.getAttribute('data-topic'), msg=>{
        if(typeof msg.value==='number') bar.set(msg.value);
      });
    });
    // timeline
    $$('.imu-timeline[data-topic]').forEach(el=>{
      const tl = new EventTimeline(el, {title: el.getAttribute('data-title')||'Events'});
      sseConnect(el.getAttribute('data-topic'), msg=>tl.push(msg));
    });
    // streaming table
    $$('.imu-stream-table[data-topic]').forEach(el=>{
      const columns = JSON.parse(el.getAttribute('data-columns')||'[]');
      const freeze  = parseInt(el.getAttribute('data-freeze')||'0',10);
      const tbl = new StreamingTable(el, {columns, freeze});
      sseConnect(el.getAttribute('data-topic'), msg=>tbl.push(msg));
    });

    applyGrid(document);
  }
  document.addEventListener('DOMContentLoaded', boot);
})();