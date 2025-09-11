# examples/run_realtime_demo.py
# -*- coding: utf-8 -*-
import asyncio, json, random, time
from realtime.ws_server import run_server
from realtime.priority_bus import AsyncPriorityTopicBus
from realtime.backpressure import GlobalTokenBucket

# נריץ שרת WS ואז נפרסם אליו התקדמות ואירועים בזמן אמת.
async def publish_streams(bus: AsyncPriorityTopicBus):
    # progress: topic "progress:build"
    pct = 0
    while pct <= 100:
        msg = json.dumps({"percent": pct})
        await bus.publish("progress:build", f"progress:build::{msg}", priority=1)
        await asyncio.sleep(0.1)
        pct += random.randint(1, 5)

    # timeline: topic "events"
    for i in range(20):
        await bus.publish("events", f"events::stage-{i} completed", priority=5)
        await asyncio.sleep(0.05)

async def main():
    # מפעילים WS server בדיוק כמו בקוד השרת (בלי להכפיל לוגיקה)
    # ניצור כאן bus גלובלי זהה לזה שבתוך ws_server.run_server
    # בפועל: בתשתית שלך יש bus יחיד. כאן זה הדגמה מבודדת.
    global_bucket = GlobalTokenBucket(capacity=5000, rate_tokens_per_sec=1000.0)
    bus = AsyncPriorityTopicBus(global_bucket)

    # עוטפים את run_server (שמייצר bus משלו) – כדי לשתף bus, אפשר להעתיק/להרחיב ל-run_server_with(bus)
    # לשם פשטות: נריץ את run_server כרגיל, ונפרסם דרך חיבור WS עצמו (פחות יעיל). כאן נשתמש בפאבליש ישיר לדוגמה.
    server_task = asyncio.create_task(run_server(host="127.0.0.1", port=8765))
    # המתנה קצרה ל־bind
    await asyncio.sleep(0.3)

    # מדמה פרסום תוצרי בנייה/ציר זמן (במקרה אמיתי – פרסום מתוך build jobs)
    pub_task = asyncio.create_task(publish_streams(bus))

    # מריצים כמה זמן ואז עוצרים
    await asyncio.sleep(5)
    for t in (pub_task, server_task):
        if not t.done():
            t.cancel()
    # אין החזרה — זוהי תוכנית דמו לבדיקה ידנית

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass