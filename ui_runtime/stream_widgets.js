/*ui_runtime/stream_widgets.js (ווידג’טים סטרימיים ללא ספריות)
/* eslint-disable */
export class WSClient {
  constructor(url) { this.url = url; this.ws = null; this.handlers = {}; }
  connect() {
    this.ws = new WebSocket(this.url);
    this.ws.onmessage = (ev) => {
      const txt = typeof ev.data === "string" ? ev.data : "";
      // פורמט אפליקטיבי פשוט: TOPIC::payload
      const idx = txt.indexOf("::");
      if (idx > 0) {
        const topic = txt.slice(0, idx);
        const payload = txt.slice(idx + 2);
        (this.handlers[topic] || []).forEach(h => h(payload));
      }
    };
  }
  sub(topic, handler) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      const int = setInterval(() => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          clearInterval(int);
          this.ws.send("SUB " + topic);
        }
      }, 50);
    } else {
      this.ws.send("SUB " + topic);
    }
    if (!this.handlers[topic]) this.handlers[topic] = [];
    this.handlers[topic].push(handler);
  }
}

export function mountProgress(el, client, topic, label) {
  el.innerHTML = `<div class="imu-progress"><div class="bar"></div><span class="lbl">${label||""}</span></div>`;
  const bar = el.querySelector(".bar");
  client.sub(topic, (payload) => {
    try {
      const o = JSON.parse(payload);
      const v = Math.max(0, Math.min(100, o.percent || 0));
      bar.style.width = v + "%";
    } catch (_) {}
  });
}

export function mountTimeline(el, client, topic, maxItems) {
  el.innerHTML = `<ul class="imu-timeline"></ul>`;
  const ul = el.querySelector("ul");
  client.sub(topic, (payload) => {
    const li = document.createElement("li");
    li.textContent = payload;
    ul.prepend(li);
    while (ul.childElementCount > (maxItems || 50)) ul.removeChild(ul.lastChild);
  });
}