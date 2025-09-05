// imu_repo/ui/web/client_widget.js  (מורחב)
(function(){
  function distinctSources(evList){ const s=new Set(); (evList||[]).forEach(ev=>{const src=ev.source||ev.url||ev.sha256||ev.kind; if(src) s.add(String(src));}); return s.size; }
  function verifyGrounded(bundle, minSources=1, minTrust=1.0){
    if(!bundle || typeof bundle.text!=="string") throw new Error("text missing");
    if(!Array.isArray(bundle.claims) || bundle.claims.length===0) throw new Error("claims missing");
    let trust=0;
    bundle.claims.forEach((c,i)=>{
      if(!c || typeof c!=="object" || !c.type || !c.text) throw new Error(`claim[${i}] bad`);
      const ev=c.evidence||[]; if(!Array.isArray(ev)||ev.length===0) throw new Error(`claim[${i}] no evidence`);
      const ds=distinctSources(ev); if(ds<minSources) throw new Error(`claim[${i}] insufficient sources`);
      let score=ds; ev.forEach(e=>{ if(e.sha256) score+=0.5; if((e.url||"").startsWith("https://")) score+=0.25; });
      trust+=score;
    });
    if(trust<minTrust) throw new Error("low total trust");
    return {trust};
  }

  class TableWidget {
    constructor(rootId, keyField){ this.root=document.getElementById(rootId); this.keyField=keyField; this.rows=new Map(); this.sortCol=null; this.sortRev=false; this.filters=new Map();}
    apply(payload){
      (payload.rows||[]).forEach(r=>{ const k=r[this.keyField]; if(k!=null) this.rows.set(String(k), r); });
      (payload.ops||[]).forEach(op=>{
        if(op.op==="upsert"){ const r=op.row||{}; const k=r[this.keyField]; if(k!=null){ const old=this.rows.get(String(k))||{}; this.rows.set(String(k), Object.assign({}, old, r)); } }
        else if(op.op==="delete"){ const k=op.key; if(k!=null) this.rows.delete(String(k)); }
      });
      this.render();
    }
    setSort(c,rev=false){ this.sortCol=c; this.sortRev=rev; this.render(); }
    setFilter(c,fn){ this.filters.set(c,fn); this.render(); }
    _data(){
      let arr=Array.from(this.rows.values());
      for(const [c,fn] of this.filters.entries()) arr=arr.filter(r=>fn(r[c]));
      if(this.sortCol) arr.sort((a,b)=>{ const va=a[this.sortCol], vb=b[this.sortCol]; return (va>vb?1:va<vb?-1:0)*(this.sortRev?-1:1); });
      return arr;
    }
    render(){
      if(!this.root) return;
      const arr=this._data();
      const cols=Array.from(new Set(arr.flatMap(o=>Object.keys(o))));
      const thead=`<thead><tr>${cols.map(c=>`<th data-col="${c}">${c}</th>`).join("")}</tr></thead>`;
      const tbody=`<tbody>${arr.map(r=>`<tr>${cols.map(c=>`<td>${(r[c]??"")}</td>`).join("")}</tr>`).join("")}</tbody>`;
      this.root.innerHTML=`<table class="tbl">${thead}${tbody}</table>`;
      this.root.querySelectorAll("th").forEach(th=>{
        th.onclick=()=>{ const c=th.getAttribute("data-col"); if(this.sortCol===c) this.sortRev=!this.sortRev; else {this.sortCol=c; this.sortRev=false;} this.render(); };
      });
    }
  }

  class ChartWidget {
    constructor(rootId, maxPoints=2048){ this.root=document.getElementById(rootId); this.maxPoints=maxPoints; this.points=[]; }
    apply(payload){
      if(payload.set){ this.points = payload.set.slice(0,this.maxPoints); }
      if(payload.append){ this.points = this.points.concat(payload.append); if(this.points.length>this.maxPoints) this.points=this.points.slice(-this.maxPoints); }
      this.render();
    }
    render(){
      if(!this.root) return;
      // רינדור פשוט ל־canvas
      if(!this.canvas){ this.canvas=document.createElement("canvas"); this.canvas.width=this.root.clientWidth||600; this.canvas.height=160; this.root.appendChild(this.canvas); }
      const ctx=this.canvas.getContext("2d"); ctx.clearRect(0,0,this.canvas.width,this.canvas.height);
      if(this.points.length<2) return;
      const xs=this.points.map(p=>p[0]), ys=this.points.map(p=>p[1]);
      const xmin=Math.min.apply(null,xs), xmax=Math.max.apply(null,xs);
      const ymin=Math.min.apply(null,ys), ymax=Math.max.apply(null,ys);
      const W=this.canvas.width, H=this.canvas.height;
      const nx=t=> (xmax===xmin? 0 : (t - xmin)/(xmax - xmin))*W;
      const ny=v=> H - (ymax===ymin? 0 : (v - ymin)/(ymax - ymin))*H;
      ctx.beginPath();
      this.points.forEach((p,i)=>{ const x=nx(p[0]), y=ny(p[1]); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
      ctx.stroke();
    }
  }

  class MetricWidget {
    constructor(rootId){ this.root=document.getElementById(rootId); this.value=null; this.unit=null; }
    apply(payload){ if(payload.value!=null) this.value=payload.value; if(payload.unit!=null) this.unit=payload.unit; this.render(); }
    render(){
      if(!this.root) return;
      this.root.textContent = (this.value==null? "-" : String(this.value)) + (this.unit? " "+this.unit : "");
    }
  }

  class LogWidget {
    constructor(rootId, maxLines=2000){ this.root=document.getElementById(rootId); this.lines=[]; this.maxLines=maxLines; }
    apply(payload){
      if(Array.isArray(payload.append)){ this.lines=this.lines.concat(payload.append); if(this.lines.length>this.maxLines) this.lines=this.lines.slice(-this.maxLines); }
      if(payload.truncate){ const n=Number(payload.truncate)||0; if(n<this.lines.length) this.lines=this.lines.slice(-n); }
      this.render();
    }
    render(){
      if(!this.root) return;
      this.root.innerHTML = this.lines.map(l=>`<div class="log ${l.lvl||"INFO"}">${(l.msg||"")}</div>`).join("");
    }
  }

  function connectWS(url, {onMsg, minSources=1, minTrust=1.0}){
    const ws = new WebSocket(url);
    ws.onopen = ()=> { ws.send(JSON.stringify({op:"ui/subscribe", bundle:{topics:["orders","grid","metrics","logs","chart"]}})); };
    ws.onmessage = (ev)=>{
      try{
        const doc = JSON.parse(ev.data);
        if(/^control\//.test(doc.op)) return;
        verifyGrounded(doc.bundle, minSources, minTrust);
        onMsg(doc);
      }catch(e){ console.warn("drop", e); }
    };
    return ws;
  }

  window.IMUClient = { TableWidget, ChartWidget, MetricWidget, LogWidget, connectWS };
})();