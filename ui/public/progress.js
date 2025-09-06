// Connect to ws://localhost:8765/stream and update progress bars + timeline
const ws = new WebSocket(`ws://${location.host}/ws/stream`);
const timeline = document.getElementById("timeline");
const bars = {};

function ensureBar(topic) {
  if (!bars[topic]) {
    const el = document.createElement("div");
    el.className = "bar";
    el.innerHTML = `<label>${topic}</label><progress value="0" max="100"></progress>`;
    document.getElementById("progress").appendChild(el);
    bars[topic]=el.querySelector("progress");
  }
}

ws.onmessage = (ev)=>{
  const msg = JSON.parse(ev.data);
  if (msg.kind === "progress") {
    ensureBar(msg.topic);
    bars[msg.topic].value = msg.value;
    const li = document.createElement("li");
    li.textContent = `[${msg.ts}] ${msg.topic}: ${msg.value}% – ${msg.note||""}`;
    timeline.appendChild(li);
  } else if (msg.kind === "event") {
    const li = document.createElement("li");
    li.textContent = `[${msg.ts}] ${msg.topic}: ${msg.note||""}`;
    timeline.appendChild(li);
  }
};