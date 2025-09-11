# imu_repo/ui/web.py
from __future__ import annotations
import json, os, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List, Optional

from obs.kpi import KPI
from optimizer.phi import compute_phi
from governance.proof_of_convergence import SafeProgressLedger
from persistence.policy_store import PolicyStore
from obs.alerts import AlertMonitor, AlertSink

DASH_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IMU Policy Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
:root { --bg:#0b0e14; --fg:#e6e6e6; --muted:#9aa0a6; --box:#141820; --acc:#4cc9f0; --ok:#00d084; --bad:#ff3366; }
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,Segoe UI,Arial;background:var(--bg);color:var(--fg)}
.header{padding:14px 18px;border-bottom:1px solid #222;display:flex;gap:12px;align-items:center}
.header h1{font-size:18px;margin:0;color:#fff}
.wrap{padding:16px;display:grid;grid-template-columns:1.1fr 1fr;gap:16px}
.box{background:var(--box);border:1px solid #1f2430;border-radius:10px;padding:14px}
h2{margin:0 0 8px 0;font-size:15px;color:#fff}
pre{white-space:pre-wrap;word-break:break-word;background:#0e121a;padding:10px;border-radius:8px;border:1px solid #1b1f2a;color:#cde}
small{color:var(--muted)}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
input[type=text],input[type=number]{background:#0e121a;color:#cde;border:1px solid #1b2030;border-radius:8px;padding:6px 8px}
button{background:#1b5e20;color:#fff;border:0;border-radius:8px;padding:8px 10px;cursor:pointer}
button.bad{background:#7f1d1d}
button.neu{background:#303c50}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.kv{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.kv div{background:#0e121a;border:1px solid #1b2030;border-radius:8px;padding:6px 8px}
.canvas-wrap{height:220px}
canvas{width:100%;height:200px;background:#0e121a;border-radius:8px;border:1px solid #1b2030}
@media (max-width: 900px){ .wrap{grid-template-columns:1fr} }
</style>
</head>
<body>
<div class="header">
  <h1>IMU Policy Dashboard</h1>
  <small id="status" style="opacity:.8">loading…</small>
</div>

<div class="wrap">
  <div class="box">
    <h2>Live Metrics</h2>
    <div class="grid2">
      <div class="kv">
        <div><small>avg (ms)</small><div id="avg">–</div></div>
        <div><small>p95 (ms)</small><div id="p95">–</div></div>
        <div><small>error rate</small><div id="err">–</div></div>
        <div><small>Φ (lower=better)</small><div id="phi">–</div></div>
      </div>
      <div class="row" style="justify-content:flex-end;gap:6px">
        <button class="neu" onclick="toggleAlerts()">Toggle Alerts</button>
        <button onclick="refreshNow()">Refresh</button>
      </div>
    </div>
    <div class="canvas-wrap"><canvas id="chart"></canvas></div>
  </div>

  <div class="box">
    <h2>Policy (thresholds & limits)</h2>
    <div class="grid2">
      <div>
        <div class="row">
          <label>max_error_rate</label>
          <input type="number" step="0.001" id="mer" value="0.02">
        </div>
        <div class="row">
          <label>max_p95_latency_ms</label>
          <input type="number" step="1" id="mp95" value="800">
        </div>
        <div class="row">
          <label>cpu_steps_max</label>
          <input type="number" step="1000" id="cpu" value="500000">
        </div>
        <div class="row">
          <label>mem_kb_max</label>
          <input type="number" step="1024" id="mem" value="65536">
        </div>
        <div class="row">
          <label>io_calls_max</label>
          <input type="number" step="10" id="io" value="10000">
        </div>
      </div>
      <div class="row" style="align-items:flex-start">
        <button onclick="savePolicy()">Save Policy</button>
        <button class="bad" onclick="reloadPolicy()">Reload</button>
      </div>
    </div>
    <small>Changes apply immediately to the active policy file.</small>
  </div>

  <div class="box">
    <h2>Ledger (tail)</h2>
    <pre id="ledger">–</pre>
  </div>

  <div class="box">
    <h2>Alerts (tail)</h2>
    <pre id="alerts">–</pre>
  </div>
</div>

<script>
let CH=[], CT=[]; // latency series
let running=true;
function $(id){ return document.getElementById(id); }

async function api(url, opts){ const r = await fetch(url, opts); if(!r.ok) throw new Error(await r.text()); return r.json(); }

async function loadPolicy(){
  const p = await api('/api/policy');
  const thr = (p.config && p.config.thresholds) || {"max_error_rate":0.02,"max_p95_latency_ms":800};
  const lim = (p.config && p.config.limits) || {"cpu_steps_max":500000,"mem_kb_max":65536,"io_calls_max":10000};
  $('mer').value = thr.max_error_rate; $('mp95').value = thr.max_p95_latency_ms;
  $('cpu').value = lim.cpu_steps_max; $('mem').value = lim.mem_kb_max; $('io').value = lim.io_calls_max;
}

async function savePolicy(){
  const cfg = {
    thresholds: {
      max_error_rate: parseFloat($('mer').value),
      max_p95_latency_ms: parseFloat($('mp95').value)
    },
    limits: {
      cpu_steps_max: parseInt($('cpu').value),
      mem_kb_max: parseInt($('mem').value),
      io_calls_max: parseInt($('io').value)
    }
  };
  await api('/api/policy', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(cfg)});
  $('status').textContent='policy saved';
}

async function reloadPolicy(){ await loadPolicy(); $('status').textContent='reloaded'; }

async function refreshNow(){
  const m = await api('/api/metrics');
  $('avg').textContent = m.kpi.avg.toFixed(2);
  $('p95').textContent = m.kpi.p95.toFixed(2);
  $('err').textContent = (m.kpi.error_rate*100).toFixed(2)+'%';
  $('phi').textContent = m.phi.toFixed(4);

  const s = await api('/api/metrics/series?tail=200');
  CH = s.series.map(p=>p.latency_ms);
  CT = s.series.map(p=>p.ts);

  const l = await api('/api/ledger?tail=20');
  $('ledger').textContent = JSON.stringify(l, null, 2);

  const a = await api('/api/alerts');
  $('alerts').textContent = JSON.stringify(a, null, 2);

  drawChart();
}

function drawChart(){
  const c = $('chart'); const g = c.getContext('2d');
  const W = c.width = c.clientWidth; const H = c.height = c.clientHeight;
  g.fillStyle = '#0e121a'; g.fillRect(0,0,W,H);
  g.strokeStyle = '#223'; g.beginPath(); for(let i=0;i<6;i++){let y=i*(H/5); g.moveTo(0,y); g.lineTo(W,y);} g.stroke();

  if(CH.length<2){ return; }
  const min = Math.min(...CH), max = Math.max(...CH);
  const pad = 10; const span = (max-min)||1;
  g.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--acc');
  g.beginPath();
  for(let i=0;i<CH.length;i++){
    const x = pad + (W-2*pad) * (i/(CH.length-1));
    const y = H - pad - (H-2*pad) * ((CH[i]-min)/span);
    if(i===0) g.moveTo(x,y); else g.lineTo(x,y);
  }
  g.stroke();
}

async function toggleAlerts(){
  running = !running;
  await api('/api/alerts/control', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({start: running})});
  $('status').textContent = running ? 'alerts: on' : 'alerts: off';
}

setInterval(refreshNow, 2000);
window.onload = async ()=>{ await loadPolicy(); await refreshNow(); };
</script>
</body>
</html>
"""

class WebUI:
    def __init__(self, host:str="0.0.0.0", port:int=8000):
        self.host=host; self.port=port
        self.kpi = KPI()
        self.ledger = SafeProgressLedger()
        self.policy = PolicyStore()
        self.alerts_sink = AlertSink()
        self.alert_monitor = AlertMonitor(self.kpi, self.policy, sink=self.alerts_sink, period_s=2.0)
        self._alerts_running = False

    def _metrics_series(self, tail:int=200) -> Dict[str,Any]:
        # מחזיר את N הרשומות האחרונות מה-KPI jsonl
        path=self.kpi.path
        data=[]
        try:
            with open(path,"r",encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try: data.append(json.loads(line))
                    except Exception: pass
        except FileNotFoundError:
            data=[]
        data=data[-tail:]
        return {"series": data}

    def serve(self, host=None, port=None):
        host = host or self.host; port = port or self.port
        ui = self

        class H(BaseHTTPRequestHandler):
            def _bytes(self, b: bytes, code: int = 200, ct: str = "text/plain; charset=utf-8"):
                self.send_response(code)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def _json(self, obj: Dict[str,Any], code:int=200):
                return self._bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"), code, "application/json; charset=utf-8")

            def do_GET(self):
                p=urlparse(self.path); q=parse_qs(p.query or "")
                if p.path == "/":
                    return self._bytes(DASH_HTML.encode("utf-8"), 200, "text/html; charset=utf-8")
                if p.path == "/api/metrics":
                    snap=ui.kpi.snapshot()
                    phi = compute_phi({"p95": snap["p95"], "latency_ms": snap["avg"], "error": snap["error_rate"]>0.0})
                    return self._json({"kpi": snap, "phi": phi})
                if p.path == "/api/metrics/series":
                    tail=int(q.get("tail",[200])[0])
                    return self._json(ui._metrics_series(tail=max(1,min(tail,5000))))
                if p.path == "/api/ledger":
                    tail=int(q.get("tail",[20])[0])
                    events=[]
                    try:
                        with open(".imu_state/ledger.jsonl","r",encoding="utf-8") as f:
                            for line in f:
                                line=line.strip()
                                if line: events.append(json.loads(line))
                        events=events[-tail:]
                    except Exception:
                        events=[]
                    return self._json({"tail": tail, "events": events})
                if p.path == "/api/alerts":
                    tail=int(q.get("tail",[50])[0])
                    events=[]
                    try:
                        with open(".imu_state/alerts.jsonl","r",encoding="utf-8") as f:
                            for line in f:
                                line=line.strip()
                                if line: events.append(json.loads(line))
                        events=events[-tail:]
                    except Exception:
                        events=[]
                    return self._json({"tail": tail, "alerts": events})
                if p.path == "/api/policy":
                    return self._json(ui.policy.current())
                self.send_response(404); self.end_headers()

            def do_POST(self):
                p=urlparse(self.path)
                ln=int(self.headers.get("Content-Length","0") or "0")
                raw=self.rfile.read(ln) if ln>0 else b"{}"
                try:
                    body=json.loads(raw.decode() or "{}")
                except Exception:
                    body={}
                # Update policy live
                if p.path == "/api/policy":
                    cur = ui.policy.current().get("config", {})
                    new_cfg = {**cur}
                    if "thresholds" in body:
                        th = new_cfg.get("thresholds", {})
                        th.update(body["thresholds"])
                        new_cfg["thresholds"]=th
                    if "limits" in body:
                        lm = new_cfg.get("limits", {})
                        lm.update(body["limits"])
                        new_cfg["limits"]=lm
                    ver = ui.policy.stage(new_cfg, note="dashboard_update")
                    ui.policy.promote(ver)
                    return self._json({"status":"ok","version":ver,"config":new_cfg})
                # alerts control
                if p.path == "/api/alerts/control":
                    start=bool(body.get("start", True))
                    if start and not ui._alerts_running:
                        ui.alert_monitor.start(); ui._alerts_running=True
                        return self._json({"running": True})
                    if (not start) and ui._alerts_running:
                        ui.alert_monitor.stop(); ui._alerts_running=False
                        return self._json({"running": False})
                    return self._json({"running": ui._alerts_running})
                self.send_response(404); self.end_headers()

        httpd = HTTPServer((host, port), H)
        print(f"[WebUI] Serving on http://{host}:{port}")
        try:
            httpd.serve_forever()
        finally:
            if self._alerts_running:
                self.alert_monitor.stop(); self._alerts_running=False

# כניסה נוחה להפעלה ישירה:
if __name__ == "__main__":
    WebUI().serve(host="127.0.0.1", port=8000)
