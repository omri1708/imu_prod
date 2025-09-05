# ui_dsl/compiler.py (הרחבה—רכיב timeline עם חיבור חי ל־run_id)
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any

def compile_ui(spec:Dict[str,Any]) -> str:
    """
    DSL דוגמתי:
    {
      "type":"page",
      "components":[
         {"type":"progress_bar","id":"p1"},
         {"type":"timeline","id":"t1","run_binding":"run_id"}
      ]
    }
    """
    head = """
<!doctype html><html><head>
<meta charset="utf-8"/>
<title>IMU UI</title>
<style>
.progress{height:16px;background:#eee;border-radius:8px;overflow:hidden}
.progress > .bar{height:100%; background:#0b74de; color:#fff; font:12px sans-serif; text-align:center}
.timeline{font:12px/1.4 sans-serif}
.timeline .evt{padding:4px 0; border-bottom:1px solid #eee}
.ts{color:#888; margin-right:8px}
</style>
<script type="module">
import {StreamTimeline} from '/static/stream_timeline.js';
window.__imu_mount = (runId) => {
  const root = document.getElementById('root');
  new StreamTimeline(root, runId);
};
</script>
</head><body><div id="root">
"""
    body = []
    for c in spec.get("components",[]):
        if c["type"]=="progress_bar":
            body.append('<div class="progress"><div data-progress class="bar" style="width:0%">0%</div></div>')
        elif c["type"]=="timeline":
            body.append('<div class="timeline" data-timeline></div>')
        else:
            body.append(f'<!-- unknown component {c["type"]} -->')
    tail = """
</div>
<script>/* runtime will call __imu_mount(runId) */</script>
</body></html>
"""
    return head + "\n".join(body) + tail