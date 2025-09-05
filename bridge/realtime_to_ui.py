# imu_repo/bridge/realtime_to_ui.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import asyncio, random, time
from realtime.tcp_framed import TCPFramedServer
from realtime.ws_push import WSPushServer
from realtime.strict_sink import StrictSink

def _mk_order_event(i: int) -> Dict[str, Any]:
    row = {"id": i, "sku": f"SKU{i%7:03d}", "qty": random.randint(1, 9), "price": round(random.uniform(5,120),2)}
    ui = {"orders_table": {"ops":[{"op":"upsert","row":row}]}}
    claims = [{
        "type": "compute",
        "text": "order stream sample",
        "evidence": [{"kind":"internal_stream","sha256":"deadbeef"*8,"source":"local"}]
    }]
    return {"text": "orders update", "claims": claims, "ui": ui}

def _mk_metrics_event(t: float, val: float) -> Dict[str, Any]:
    ui = {
        "qps_metric": {"value": round(val,2), "unit": "req/s"},
        "latency_chart": {"append": [[t, 100 + 50*random.random()]]},
        "logs_panel": {"append":[{"lvl":"INFO","msg":f"t={round(t,2)} val={round(val,2)}"}]},
    }
    claims = [{
        "type":"telemetry","text":"live metrics",
        "evidence":[{"kind":"internal_metrics","source":"local"}]
    }]
    return {"text":"metrics tick", "claims":claims, "ui": ui}

async def handler(op: str, bundle: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    # ניהול subscribe "לוגי": לא צריך state פר־נושא בקוד הדוגמה
    if op == "ui/subscribe":
        return "control/ack", {"ok": True, "topics": bundle.get("topics", [])}
    return "control/ack", {"ok": True}

async def run_push_servers():
    policy = {"min_distinct_sources": 1, "min_total_trust": 1.0, "perf_sla":{"latency_ms":{"p95_max":200}}}
    sink = StrictSink(policy)

    # TCP (למי שרוצה לקוח TCP)
    tcp = TCPFramedServer("127.0.0.1", 9401, handler, sink)
    await tcp.start()

    # WS Push — נרשום חיבורים ונשדר לכולם
    ws = WSPushServer("127.0.0.1", 9402, handler, sink)
    await ws.start()

    async def publisher():
        i = 0
        while True:
            i += 1
            t = time.time()
            # אירועי הזמנה
            await ws.broadcast("ui/update", _mk_order_event(i))
            # טלמטריה
            await ws.broadcast("ui/update", _mk_metrics_event(t, val=5+random.random()*3))
            await asyncio.sleep(0.5)

    async def serve():
        await ws.run_forever()

    print("Starting realtime bridge: TCP 9401, WS 9402")
    await asyncio.gather(serve(), publisher())

if __name__ == "__main__":
    asyncio.run(run_push_servers())


#TODO-
# הערה: לגרסת Push אמיתית לכל הלקוחות ב־WS צריך לשמור רשימת חיבורים פתוחים ולשגר אליהם 
# pack("ui/update", bundle) עם ui: {...} — 
# המבנה אצלנו כבר Grounded-Strict, כך שה־StrictSink בצד השרת יאשר אותו. את זה ניתן להוסיף בקובץ זה ע״י ניהול 
# set של חיבורים; שמרתי את הקוד קצר וברור — אם תרצה, אוסיף כאן גרסת Push מלאה.