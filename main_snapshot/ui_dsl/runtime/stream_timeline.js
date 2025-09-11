ui_dsl/runtime/stream_timeline.js
// Minimal SSE client that binds to /events/:run_id and updates Progress & Timeline components
export class StreamTimeline {
  constructor(rootEl, runId) {
    this.root = rootEl;
    this.runId = runId;
    this.events = [];
    this.progressEl = this.root.querySelector("[data-progress]");
    this.timelineEl = this.root.querySelector("[data-timeline]");
    this._connect();
  }
  _connect() {
    const ev = new EventSource(`/events/${this.runId}`);
    ev.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data);
        this.events.push(payload);
        this._render();
      } catch (_) {}
    };
    ev.addEventListener("start", e => {
      const p = JSON.parse(e.data); this.events.push(p); this._render();
    });
    ev.addEventListener("finished", e => {
      const p = JSON.parse(e.data); this.events.push(p); this._render(100);
    });
    ev.addEventListener("error", e => {
      const p = JSON.parse(e.data); this.events.push(p); this._render();
    });
  }
  _render(forceProgress) {
    const last = this.events[this.events.length-1] || {};
    const pct = typeof forceProgress === "number" ? forceProgress :
                last.progress ?? Math.min(95, this.events.length * 10);
    if (this.progressEl) {
      this.progressEl.style.width = `${pct}%`;
      this.progressEl.textContent = `${pct}%`;
    }
    if (this.timelineEl) {
      this.timelineEl.innerHTML = this.events.map(ev => {
        const t = new Date(ev.ts*1000).toLocaleTimeString();
        return `<div class="evt"><span class="ts">${t}</span> <code>${ev.type}</code> ${ev.adapter||""} ${ev.stage||""}</div>`;
      }).join("");
    }
  }
}