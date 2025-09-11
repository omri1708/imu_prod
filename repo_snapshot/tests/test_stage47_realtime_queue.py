# imu_repo/tests/test_stage47_realtime_queue.py
from __future__ import annotations
import asyncio, random, time
from engine.realtime_and_dist import MicroRuntime

async def handler(payload):
    # מדמה פעולה קצרה עם כשל אקראי נמוך
    await asyncio.sleep(0.002)
    if random.random() < 0.02:
        raise RuntimeError("flaky")
    return {"ok": True}

async def main():
    rt = MicroRuntime()
    # מציף את התור
    for i in range(200):
        await rt.submit({"n": i})

    # מפעיל שני וורקרים
    w1 = asyncio.create_task(rt.worker("svc", handler))
    w2 = asyncio.create_task(rt.worker("svc", handler))

    # רוטינת תחזוקה: requeue ל-inflight תקוע
    async def maint():
        for _ in range(30):
            moved = rt.q.requeue_stale(0.2)  # קצר כדי לבדוק שהמנגנון עובד
            await asyncio.sleep(0.05)

    m = asyncio.create_task(maint())

    # מחכים קצת עד סיום עיבוד
    await asyncio.sleep(2.5)
    # עוצרים וורקרים
    for t in (w1, w2, m):
        t.cancel()
    return 0

if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("OK")
        raise SystemExit(0)
    except Exception as e:
        print("FAIL", e)
        raise SystemExit(1)