// imu_repo/ui/proofs_view.js
(function(){
  function el(id){ return document.getElementById(id); }
  async function start(){
    const out = el("out");
    const btn = el("send");
    const inp = el("msg");

    const ws = new WebSocket("ws://127.0.0.1:8766/rt");
    ws.onopen = ()=> out.textContent += "[open]\n";
    ws.onclose= ()=> out.textContent += "[close]\n";
    ws.onmessage = (ev)=>{
      // ייתכן prefix של eco-id, ננסה לפצל ב-"|"
      let data = ev.data;
      if (typeof data === "string"){
        const p = data.indexOf("|");
        if (p>0){ data = data.slice(p+1); }
        try{
          const obj = JSON.parse(data);
          out.textContent += "TEXT: " + (obj.text||"") + "\n";
          out.textContent += "CLAIMS:\n";
          for(const c of (obj.claims||[])){
            out.textContent += " - " + c.digest + " trust>=" + c.min_trust + "\n";
          }
          out.textContent += "SIG: " + (obj.sig?obj.sig.alg:"(none)") + "\n\n";
        }catch(e){
          out.textContent += "RAW: " + data + "\n";
        }
      }else{
        out.textContent += "[binary]\n";
      }
    };
    btn.onclick = ()=> {
      ws.send(inp.value || "hello");
    };
  }
  if (document.readyState==="complete" || document.readyState==="interactive"){
    start();
  }else{
    document.addEventListener("DOMContentLoaded", start);
  }
})();