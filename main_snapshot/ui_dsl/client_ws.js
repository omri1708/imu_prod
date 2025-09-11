// ui_dsl/client_ws.js
// IMU_WS: קליינט יחיד עם subscribe(RegExp|String, fn)
(function(global){
  const subs = [];
  let ws = null, retry = 200;
  function connect(){
    const url = (global.IMU_WS_URL) || ("ws://"+location.hostname+":8766/ws");
    ws = new WebSocket(url);
    ws.onopen = ()=>{ retry=200; };
    ws.onmessage = (ev)=>{
      try{
        const msg = JSON.parse(ev.data);
        const t = msg.topic || "";
        for(const [pat,fn] of subs){
          const ok = (pat instanceof RegExp) ? pat.test(t) : (pat===t);
          if(ok) try{ fn(msg); }catch(e){ console.error(e); }
        }
      }catch(e){ console.error("bad message", e); }
    };
    ws.onclose = ()=>{ setTimeout(connect, Math.min(5000, retry)); retry*=1.5; };
    ws.onerror = ()=>{ try{ws.close();}catch{} };
  }
  connect();
  global.IMU_WS = {
    subscribe(pat, fn){ subs.push([pat, fn]); }
  };
})(window);