// ui_dsl/components/streams.ts
// קליינט WebSocket עם back-pressure ו-priority queues.

type Priority = 0 | 1 | 2; // 0=high,1=normal,2=low

export interface StreamEvent {
  topic: string;
  ts: number;          // epoch ms
  type: string;        // "progress" | "log" | "metric" | "timeline"
  payload: any;
}

export class PriorityQueue<T> {
  private q: Map<Priority, T[]> = new Map([[0,[]],[1,[]],[2,[]]]);
  enqueue(item: T, p: Priority=1) { this.q.get(p)!.push(item); }
  dequeue(): T | undefined {
    for (const p of [0,1,2] as Priority[]) {
      const arr = this.q.get(p)!;
      if (arr.length) return arr.shift();
    }
    return undefined;
  }
  size(): number { return ([...this.q.values()].reduce((a,b)=>a+b.length,0)); }
}

export interface WSConfig {
  url: string;                   // ws://host/stream?topic=...
  burstLimit: number;            // N - מספר מירבי בבת אחת
  globalRatePerSec: number;      // קצב כולל
}

export class StreamClient {
  private ws?: WebSocket;
  private q = new PriorityQueue<StreamEvent>();
  private sentThisSecond = 0;
  private lastTick = Date.now();
  constructor(private cfg: WSConfig) {}

  connect(onEvent: (ev: StreamEvent)=>void) {
    this.ws = new WebSocket(this.cfg.url);
    this.ws.onmessage = (m) => {
      try {
        const ev: StreamEvent = JSON.parse(m.data);
        onEvent(ev);
      } catch {}
    };
    // משאבת back-pressure לשידורים החוצה (אם צריך לשלוח ack/commands)
    setInterval(()=>this.pump(), 50);
    setInterval(()=>{ this.sentThisSecond = 0; this.lastTick = Date.now(); }, 1000);
  }

  sendCommand(topic: string, type: string, payload: any, p: Priority=1) {
    this.q.enqueue({topic, ts: Date.now(), type, payload}, p);
  }

  private pump() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    let n = 0;
    while (this.q.size() && n < this.cfg.burstLimit && this.sentThisSecond < this.cfg.globalRatePerSec) {
      const ev = this.q.dequeue()!;
      this.ws.send(JSON.stringify(ev));
      n++; this.sentThisSecond++;
    }
  }
}