// imu_repo/ui/web/client_widget.js
(function(){
  function distinctSources(evList){
    const seen = new Set();
    for(const ev of evList||[]){
      const src = ev.source || ev.url || ev.sha256 || ev.kind;
      if(src) seen.add(String(src));
    }
    return seen.size;
  }
  function verifyGrounded(bundle, minSources=1, minTrust=1.0){
    if(!bundle || typeof bundle.text!=="string") throw new Error("text missing");
    if(!Array.isArray(bundle.claims) || bundle.claims.length===0) throw new Error("claims missing");
    let trust = 0.0;
    bundle.claims.forEach((c,i)=>{
      if(typeof c!=="object" || !c) throw new Error(`claim[${i}] bad`);
      if(!c.type || !c.text) throw new Error(`claim[${i}] core`);
      const ev = c.evidence||[];
      if(!Array.isArray(ev) || ev.length===0) throw new Error(`claim[${i}] no evidence`);
      const ds = distinctSources(ev);
      if(ds<minSources) throw new Error(`claim[${i}] insufficient sources`);
      let score=ds;
      ev.forEach(e=>{
        if(e.sha256) score += 0.5;
        if((e.url||"").startsWith("https://")) score += 0.25;
      });
      trust += score;
    });
    if(trust<minTrust) throw new Error("low total trust");
    return {trust};
  }

  // וידג’ט טבלה בסיסי (רינדור לדום), עם מיון/סינון לקוח (בתמצית)
  class TableWidget {
    constructor(rootId, keyField){
      this.root = document.getElementById(rootId);
      this.keyField = keyField;
      this.rows = new Map();
      this.sortCol = null; this.sortRev = false;
      this.filters = new Map();
    }
    apply(payload){
      const ops = payload.ops||[];
      const rows = payload.rows||[];
      rows.forEach(r=>{ if(r[this.keyField]!=null) this.rows.set(String(r[this.keyField]), r); });
      ops.forEach(op=>{
        if(op.op==="upsert"){
          const r = op.row||{};
          const k = r[this.keyField];
          if(k!=null){
            const old = this.rows.get(String(k))||{};
            this.rows.set(String(k), Object.assign({}, old, r));
          }
        }else if(op.op==="delete"){
          const k=op.key; if(k!=null) this.rows.delete(String(k));
        }
      });
      this.render();
    }
    setSort(col, rev=false){ this.sortCol=col; this.sortRev=rev; this.render(); }
    setFilter(col, fn){ this.filters.set(col, fn); this.render(); }
    _filteredSorted(){
      let arr = Array.from(this.rows.values());
      for(const [col,fn] of this.filters.entries()){
        arr = arr.filter(r=>fn(r[col]));
      }
      if(this.sortCol){
        arr.sort((a,b)=>{
          const va=a[this.sortCol], vb=b[this.sortCol];
          return (va>vb?1:va<vb?-1:0)*(this.sortRev?-1:1);
        });
      }
      return arr;
    }
    render(){
      if(!this.root) return;
      const arr = this._filteredSorted();
      const cols = Array.from(new Set(arr.flatMap(o=>Object.keys(o))));
      const thead = `<thead><tr>${cols.map(c=>`<th data-col="${c}">${c}</th>`).join("")}</tr></thead>`;
      const tbody = `<tbody>${arr.map(r=>`<tr>${cols.map(c=>`<td>${(r[c]??"")}</td>`).join("")}</tr>`).join("")}</tbody>`;
      this.root.innerHTML = `<table class="tbl">${thead}${tbody}</table>`;
      // האזנה למיון
      this.root.querySelectorAll("th").forEach(th=>{
        th.onclick = ()=>{
          const c = th.getAttribute("data-col");
          if(this.sortCol===c){ this.sortRev = !this.sortRev; } else { this.sortCol=c; this.sortRev=false; }
          this.render();
        };
      });
    }
  }

  function connectWS(url, {onMsg, minSources=1, minTrust=1.0}){
    const ws = new WebSocket(url);
    ws.onopen = ()=> {
      ws.send(JSON.stringify({op:"ui/subscribe", bundle:{topics:["orders","grid"]}}));
    };
    ws.onmessage = (ev)=>{
      try{
        const doc = JSON.parse(ev.data);
        if(doc.op==="control/hello") return;
        if(doc.op==="control/error"){ console.warn("server error", doc.bundle); return; }
        // Grounded-Strict בצד לקוח
        verifyGrounded(doc.bundle, minSources, minTrust);
        onMsg(doc);
      }catch(e){
        console.warn("drop message:", e);
      }
    };
    return ws;
  }

  window.IMUClient = { TableWidget, connectWS };
})();