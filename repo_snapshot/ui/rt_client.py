# imu_repo/ui/rt_client.py
from __future__ import annotations
import asyncio, struct, json
from typing import Dict, Any
from realtime.protocol import pack, unpack
from ui.dsl_runtime_rt import UISession, TableWidget, GridWidget

class RTClientTCP:
    def __init__(self, host: str, port: int, ui: UISession):
        self.host = host
        self.port = port
        self.ui = ui

    async def _send(self, writer: asyncio.StreamWriter, op: str, bundle: Dict[str, Any]) -> None:
        data = pack(op, bundle)
        writer.write(struct.pack(">I", len(data)))
        writer.write(data)
        await writer.drain()

    async def run(self):
        reader, writer = await asyncio.open_connection(self.host, self.port)
        # hello
        hdr = await reader.readexactly(4)
        n = struct.unpack(">I", hdr)[0]
        raw = await reader.readexactly(n)
        _op, _bundle = unpack(raw)  # control/hello
        # נרשם לערוץ עדכונים (לפי פרוטוקול לוגי פשוט)
        await self._send(writer, "ui/subscribe", {"topics": ["orders", "grid"]})
        # לולאת קבלה
        while True:
            try:
                hdr = await reader.readexactly(4)
                n = struct.unpack(">I", hdr)[0]
                raw = await reader.readexactly(n)
            except asyncio.IncompleteReadError:
                break
            op, bundle = unpack(raw)
            if op == "control/error":
                # ניתן ללוגג; כאן פשוט מדפיסים
                print("ERROR:", bundle)
                continue
            # Grounded-Strict בצד הלקוח (UISession יאמת ויעדכן ווידג’טים)
            self.ui.handle_stream_message({"op": op, "bundle": bundle})

async def demo():
    ui = UISession(min_sources=1, min_trust=1.0)
    ui.register("orders_table", TableWidget(key_field="id"))
    ui.register("main_grid", GridWidget())
    client = RTClientTCP("127.0.0.1", 9401, ui)
    await client.run()

if __name__ == "__main__":
    asyncio.run(demo())
