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


def _claims(tag: str) -> list[dict]:
    return [{
        "type": "telemetry",
        "text": f"{tag} stream",
        "evidence": [{"kind":"internal_stream","source":"local"}]
    }]

def ev_orders(i: int) -> Dict[str, Any]:
    row = {"id": i, "sku": f"SKU{i%7:03d}", "qty": random.randint(1,9), "price": round(random.uniform(5,120),2)}
    return {"text":"orders update", "claims": _claims("orders"),
            "ui":{"orders_table":{"ops":[{"op":"upsert","row":row}]}}}

def ev_metrics(t: float) -> Dict[str, Any]:
    return {"text":"metrics tick", "claims": _claims("metrics"),
            "ui":{"qps_metric":{"value":round(5+random.random()*3,2), "unit":"req/s"},
                  "latency_chart":{"append":[[t, 100+50*random.random()]]},
                  "logs_panel":{"append":[{"lvl":"INFO","msg":f"tick {round(t,2)}"}]}}}

def ev_heatbar() -> Dict[str, Any]:
    # Heatmap אקראי + עדכון ברים
    updates = [[random.randint(0,7), random.randint(0,7), random.random()]]
    bars = [["A", random.random()*5], ["B", random.random()*3], ["C", random.random()*7]]
    return {"text":"viz update","claims":_claims("viz"),
            "ui":{"heatmap":{"inc":updates},
                  "barchart":{"set":bars}}}


async def handler(op: str, bundle: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if op == "ui/subscribe":
        return "control/ack", {"ok": True, "topics": bundle.get("topics", [])}
    return "control/ack", {"ok": True}

async def run_push_servers():
    policy = {"min_distinct_sources":1, "min_total_trust":1.0, "perf_sla":{"latency_ms":{"p95_max":200}}}
    sink = StrictSink(policy)

    tcp = TCPFramedServer("127.0.0.1", 9401, handler, sink)
    await tcp.start()

    ws = WSPushServer("127.0.0.1", 9402, handler, sink,
                      queue_max=512, msg_rate=200, byte_rate=2_000_000, burst_msgs=400, burst_bytes=4_000_000)
    await ws.start()

    async def publisher():
        i=0
        while True:
            i+=1; t=time.time()
            await ws.broadcast("ui/update", ev_orders(i), topic="orders")
            await ws.broadcast("ui/update", ev_metrics(t), topic="metrics")
            await ws.broadcast("ui/update", ev_heatbar(), topic="viz")
            await asyncio.sleep(0.25)

    print("Starting realtime bridge: TCP 9401, WS 9402 (topics: orders, metrics, viz)")
    await asyncio.gather(ws.run_forever(), publisher())

if __name__ == "__main__":
    asyncio.run(run_push_servers())


#TODO-
# הערה: לגרסת Push אמיתית לכל הלקוחות ב־WS צריך לשמור רשימת חיבורים פתוחים ולשגר אליהם 
# pack("ui/update", bundle) עם ui: {...} — 
# המבנה אצלנו כבר Grounded-Strict, כך שה־StrictSink בצד השרת יאשר אותו. את זה ניתן להוסיף בקובץ זה ע״י ניהול 
# set של חיבורים; שמרתי את הקוד קצר וברור — אם תרצה, אוסיף כאן גרסת Push מלאה.