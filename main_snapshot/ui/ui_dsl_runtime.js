// ui/ui_dsl_runtime.js
export function connectProgress(topic, onEvent) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host.replace(/:\d+$/,'')}:8765/stream/${encodeURIComponent(topic)}`;
  const ws = new WebSocket(url);
  ws.onmessage = (ev)=> {
    const msg = JSON.parse(ev.data);
    onEvent && onEvent(msg);
  };
  return ws;
}

// דוגמת שימוש: עדכון progress bar ו־event timeline
export function attachProgressUI(topic) {
  const bar = document.querySelector("#progress");
  const timeline = document.querySelector("#timeline");
  connectProgress(topic, (msg)=>{
    if (msg.kind === "progress") {
      bar.value = msg.data.pct || 0;
      bar.dataset.phase = msg.data.phase || "";
    }
    const li=document.createElement("li");
    li.textContent = `[${new Date(msg.ts*1000).toLocaleTimeString()}] ${msg.kind}: ${JSON.stringify(msg.data)}`;
    timeline.prepend(li);
  });
}