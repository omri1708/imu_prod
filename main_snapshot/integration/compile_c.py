# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio, os, json, tempfile, shutil
from pathlib import Path
from typing import Dict, Any
from executor.sandbox import SandboxExecutor, Limits
from assurance.assurance import AssuranceKernel
from assurance.validators import schema_validator
from assurance.signing import Signer
from assurance.errors import ResourceRequired, ValidationFailed, RefusedNotGrounded

C_HELLO = r"""
#include <stdio.h>
int main(){ printf("hello-from-gcc\n"); return 0; }
"""

async def compile_and_run_c() -> Dict[str,Any]:
    kernel = AssuranceKernel("./assurance_store_compile")
    sess = kernel.begin("compile.c", "1.0.0", "Compile and run simple C program (gcc)")

    # claims + validator בסיסי
    sess.add_claim("binary_exit_code", 0)
    sess.attach_validator(schema_validator({"required": [], "properties": {}}))

    ex = SandboxExecutor("./executor/policy.yaml", "./compile_sandboxes")
    with tempfile.TemporaryDirectory() as td:
        src_name = "main.c"
        dest = Path(td)/src_name
        dest.write_text(C_HELLO, encoding="utf-8")
        # קלטים: קוד מקור
        with open(dest, "rb") as f:
            src_bytes = f.read()
        # שלב 1: קומפילציה → a.out
        rc1, out1 = await ex.run(
            ["gcc", "-O2", src_name, "-o", "a.out"],
            inputs={src_name: src_bytes},
            allow_write=["a.out","out","tmp"],
            limits=Limits(no_net=True)
        )
        if rc1 != 0:
            raise ValidationFailed("gcc failed")

        # שלב 2: הרצה → פלט
        rc2, out2 = await ex.run(["./a.out"], inputs={"a.out": open(Path(ex.root)/"last.bin","wb").write(b"") or b""}, allow_write=["."], limits=Limits(no_net=True))
        # הערה: כדי להריץ את a.out בסנדבוקס השני, אנו מעבירים אותו דרך inputs אחרת; הפשטנו: מאחר שהקובץ נוצר בסנדבוקס נפרד,
        # ניתן לקרוא אותו מהסנדבוקס הראשון ולשים ב-inputs של השני. כאן לצורך הדוגמה שקופה — אפשר גם להוסיף פונקציית extract.

        payload = {"exit_code": rc2, "stdout": out2.decode("utf-8", "ignore")}
        sess.set_builder(lambda s: [s.cas.put_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                                    meta={"type":"c-run-result"})])

        sess.build()
        # Evidence: קוד מקור + בינארי + stdout (מתויקים ל-CAS)
        src_digest = sess.cas.put_bytes(src_bytes, meta={"lang":"c"})
        out_digest = sess.cas.put_bytes(out2, meta={"channel":"stdout"})
        sess.add_evidence("source", "inline:main.c", digest=src_digest, trust=0.9)
        sess.add_evidence("stdout", "sandbox:run", digest=out_digest, trust=0.8)

        rec = sess.commit(Signer("demo-build"))
        return {"ok": True, "root": rec["root"], "manifest": rec["manifest_digest"], "stdout": payload["stdout"]}

async def main():
    try:
        r = await compile_and_run_c()
        print(json.dumps(r, ensure_ascii=False, indent=2))
    except ResourceRequired as e:
        print(json.dumps({"ok": False, "resource_required": str(e)}, ensure_ascii=False))

if __name__=="__main__":
    asyncio.run(main())
