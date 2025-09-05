// ui_dsl/runtime/client.js
export function connectTimeline(onEvent, userId="anon") {
  const src = new EventSource(`/events?topic=timeline`, { withCredentials:false });
  src.onmessage = (evt)=> {
    try { const data = JSON.parse(evt.data); onEvent(data); }
    catch(e){ /* ignore */ }
  };
  src.onerror = ()=> { /* אפשר רה-קונקט */ };
  return ()=>src.close();
}