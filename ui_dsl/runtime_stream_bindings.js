export class StreamClient {
  constructor(baseUrl, topic) {
    this.baseUrl = baseUrl;
    this.topic = topic;
  }
  async pollOnce(maxItems=100) {
    const r = await fetch(`${this.baseUrl}/stream/poll?topic=${encodeURIComponent(this.topic)}&max=${maxItems}`);
    if (!r.ok) throw new Error("stream poll failed");
    return await r.json();
  }
}

// דוגמה: עידכון ProgressBar/Timeline בלקוח
export async function attachProgress(ui, client, updateIntervalMs=500) {
  async function tick() {
    try {
      const events = await client.pollOnce(100);
      for (const ev of events) {
        if (ev.evt === "start") ui.timeline.push({t:ev.ts_ms, label:"start"});
        if (ev.evt === "policy_violation") ui.timeline.push({t:ev.ts_ms, label:"policy_violation"});
        if (ev.evt === "finish") ui.timeline.push({t:ev.ts_ms, label:"finish"});
        if (ev.progress && ui.progress) ui.progress.value = ev.progress;
      }
    } catch(e) { /* אפשר לרשום לוג */ }
    setTimeout(tick, updateIntervalMs);
  }
  tick();
}