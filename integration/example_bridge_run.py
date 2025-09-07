# -*- coding: utf-8 -*-
import asyncio, json, sys
from integration.bridge_exec import run_command_with_assurance

async def main():
    cmd = ["echo", "hello-assurance"]
    res = await run_command_with_assurance(cmd)
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__=="__main__":
    asyncio.run(main())
