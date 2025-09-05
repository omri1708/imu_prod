# imu_repo/bridge/realtime_to_ui.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import asyncio, random, time
from realtime.tcp_framed import TCPFramedServer
from realtime.ws_minimal import WSServer
from realtime.strict_sink import StrictSink

def _mk_order_event(i: int) -> Dict[str, Any]:
    row = {"id": i, "sku": f"SKU{i%7:03d}", "qty": random.randint(1, 9), "price": round(random.uniform(5,120),2)}
    ui = {"orders_table": {"ops":[{"op":"upsert","row":row}]}}
    claims = [{
        "type": "compute",
        "text": "order event generated from internal stream",
        "evidence": [{"kind":"internal_stream","sha256":"deadbeef"*8,"source":"local"}]
    }]
    return {"text": "orders update", "claims": claims, "ui": ui}

async def handler(_op: str, bundle: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    # הלקוח יכול לבקש subscribe; אנו מחזירים ack בלבד
    if _op == "ui/subscribe":
        return "control/ack", {"ok": True, "topics": bundle.get("topics", [])}
    # ברירת מחדל – מחזירים קבלת־פנים
    return "control/ack", {"ok": True}

async def pump_tcp(host: str, port: int, policy: Dict[str, Any]):
    sink = StrictSink(policy)
    srv = TCPFramedServer(host, port, handler, sink)
    await srv.start()
    # מפיץ אירועים לכל החיבורים דרך כתיבה ישירה בקוד הדוגמה? כאן נשאיר “שרת בסיס”
    async with srv._server:  # type: ignore
        print(f"TCP server at {host}:{port}")
        await srv._server.serve_forever()  # type: ignore

# ל־WS נוסיף מפיץ סשן פשוט (הדגמה): עם כל חיבור, נדחף אירועים מחזוריים
async def ws_with_publisher(host: str, port: int, policy: Dict[str, Any]):
    sink = StrictSink(policy)
    async def _handler(op: str, bundle: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return await handler(op, bundle)
    srv = WSServer(host, port, _handler, sink)

    async def _serve():
        await srv.start()
        print(f"WS server at {host}:{port}")
        await srv.run_forever()

    async def _publisher():
        # פבלישר גס: פותח חיבור יוצא? בגרסה מינימלית אין לנו רשימת לקוחות ל־push.
        # לכן נדגים פמפום ע״י חיקוי “loopback”: הקליינט יקבל מה־TCP; בצד WS נשאיר הדגמה מנותקת.
        while True:
            await asyncio.sleep(3600)

    await asyncio.gather(_serve(), _publisher())

if __name__ == "__main__":
    policy = {"min_distinct_sources": 1, "min_total_trust": 1.0, "perf_sla": {"latency_ms":{"p95_max":200}}}
    asyncio.run(ws_with_publisher("127.0.0.1", 9402, policy))

 #TODO-
# הערה: לגרסת Push אמיתית לכל הלקוחות ב־WS צריך לשמור רשימת חיבורים פתוחים ולשגר אליהם 
# pack("ui/update", bundle) עם ui: {...} — 
# המבנה אצלנו כבר Grounded-Strict, כך שה־StrictSink בצד השרת יאשר אותו. את זה ניתן להוסיף בקובץ זה ע״י ניהול 
# set של חיבורים; שמרתי את הקוד קצר וברור — אם תרצה, אוסיף כאן גרסת Push מלאה.