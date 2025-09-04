# imu_repo/core/vm/vm.py
from __future__ import annotations
import time, threading
from typing import Any, Dict, List, Tuple, Optional, Callable

# אכיפת Grounding קשיח
from grounded.fact_gate import FactGate
from grounded.validators import ValidationError as GroundValidationError

# סנדבוקסים (רשת/קבצים) — קיימים בריפו
from adapters.fs_sandbox import FSSandbox
from adapters.net_sandbox import NetSandbox

class VMError(Exception): ...
class VMHalt(Exception): ...

class Limits:
    def __init__(self, cpu_steps_max=500_000, mem_kb_max=64*1024, io_calls_max=10_000,
                 max_async_tasks=16, max_sleep_ms=5_000):
        self.cpu_steps_max = int(cpu_steps_max)
        self.mem_kb_max = int(mem_kb_max)
        self.io_calls_max = int(io_calls_max)
        self.max_async_tasks = int(max_async_tasks)
        self.max_sleep_ms = int(max_sleep_ms)

class TaskResult:
    def __init__(self, ok: bool, code: int, body: Dict[str,Any], ctx: Dict[str,Any], err: Optional[str]=None):
        self.ok = ok; self.code = code; self.body = body; self.ctx = ctx; self.err = err

class VM:
    """
    VM מבוסס־מחסנית + טבלת תוויות (Labels), עם פריימים/קריאות־משנה, Heap, סנדבוקס IO, Async.
    """
    def __init__(self, limits: Optional[Limits]=None):
        self.limits = limits or Limits()

    # ---------- Program execution ----------
    def run(self, program: List[Dict[str,Any]], ctx: Optional[Dict[str,Any]]=None) -> Tuple[int, Dict[str,Any], Dict[str,Any]]:
        """
        מחזיר (status_code, body, ctx)
        """
        ctx = ctx or {}
        # ניטור משאבים
        steps = 0
        io_calls = 0
        # רגיסטרים, מחסנית, Heap, קריאות־משנה
        registers: Dict[str,Any] = {}
        stack: List[Any] = []
        heap: Dict[int, Any] = {}   # oid -> dict/list/primitive
        next_oid = 1

        # Async
        tasks: Dict[str,TaskResult] = {}
        task_threads: Dict[str, threading.Thread] = {}

        # בניית טבלת תוויות (Labels) ופונקציות
        label_to_pc: Dict[str,int] = {}
        func_bounds: Dict[str,Tuple[int,int]] = {}
        for i, ins in enumerate(program):
            op = ins.get("op")
            if op == "LABEL":
                nm = ins.get("name")
                if nm: label_to_pc[nm] = i
            elif op == "FUNC":
                nm = ins.get("name"); end = ins.get("end")  # ENDFUNC pc ימולא בפרקדור פרה־פס
                if nm: func_bounds[nm] = (i, -1)
        # השלם end של פונקציות
        last_func = None
        for i, ins in enumerate(program):
            if ins.get("op") == "FUNC":
                last_func = ins.get("name")
            elif ins.get("op") == "ENDFUNC":
                if last_func and last_func in func_bounds:
                    s,_ = func_bounds[last_func]
                    func_bounds[last_func] = (s, i)
                    last_func = None

        # מחסנית קריאות: (ret_pc, saved_registers)
        callstack: List[Tuple[int, Dict[str,Any]]] = []

        # שים אובייקטים עזר ב־ctx
        ctx.setdefault("__claims__", [])
        ctx.setdefault("__tasks__", {})
        ctx.setdefault("__registers__", registers)
        ctx.setdefault("__heap__", heap)

        def mem_kb_estimate() -> int:
            # אומדן גס — ספירת איברים; במערכות אמת תרצה tracer/objgraph
            count = 0
            for v in heap.values():
                if isinstance(v, dict): count += len(v)*4
                elif isinstance(v, list): count += len(v)*2
                else: count += 1
            return count # "יחידות" — מספיק כדי לסמן חריגות בקירוב

        pc = 0
        code, body = 200, {}
        N = len(program)

        # עזר: רזולוציית ערך "reg:NAME"
        def resolve(v):
            if isinstance(v, str) and v.startswith("reg:"):
                return registers.get(v.split(":",1)[1])
            return v

        # פקודות עזר
        def push(x): stack.append(x)
        def pop() -> Any:
            if not stack: raise VMError("stack_underflow")
            return stack.pop()

        # Async worker
        def _run_subtask(tid: str, subprog: List[Dict[str,Any]], subctx: Dict[str,Any]):
            try:
                st = time.time()
                c, b, cctx = self.run(subprog, subctx)
                tasks[tid] = TaskResult(True, c, b, cctx, None)
            except Exception as e:
                tasks[tid] = TaskResult(False, 500, {"error":"task_failed","detail":str(e)}, subctx, str(e))

        # לולאת ביצוע
        while pc < N:
            ins = program[pc]
            op = ins.get("op")
            steps += 1
            if steps > self.limits.cpu_steps_max:
                raise VMError("cpu_steps_exceeded")

            # ---- אריתמטיקה/זיכרון בסיסיים ----
            if op == "PUSH":
                push(ins.get("value"))
                pc += 1
            elif op == "POP":
                pop(); pc += 1
            elif op == "LOAD":  # LOAD var -> דוחף ערך רגיסטר למחסנית
                push(registers.get(ins.get("reg")))
                pc += 1
            elif op == "STORE": # STORE reg <- pop()
                registers[ins.get("reg")] = pop()
                pc += 1
            elif op == "ADD":
                b=pop(); a=pop(); push((a or 0)+(b or 0)); pc += 1
            elif op == "SUB":
                b=pop(); a=pop(); push((a or 0)-(b or 0)); pc += 1
            elif op == "MUL":
                b=pop(); a=pop(); push((a or 0)*(b or 0)); pc += 1
            elif op == "DIV":
                b=pop(); a=pop()
                if b == 0:
                    raise VMError("div_by_zero")
                push((a or 0)/(b or 0)); pc += 1

            # ---- השוואות/קפיצות ----
            elif op == "CMP":  # CMP a,b -> דוחף -1/0/1
                b=pop(); a=pop()
                push(-1 if a<b else (1 if a>b else 0)); pc += 1
            elif op == "JUMP":
                tgt = ins.get("to"); 
                if isinstance(tgt, int): pc = tgt
                else: pc = label_to_pc.get(str(tgt), pc+1)
            elif op == "JZ":
                cond = pop()
                if cond == 0 or cond is False or cond is None:
                    tgt = ins.get("to"); pc = (label_to_pc.get(str(tgt), pc+1) if not isinstance(tgt,int) else tgt)
                else:
                    pc += 1
            elif op == "JNZ":
                cond = pop()
                if cond != 0 and cond is not False and cond is not None:
                    tgt = ins.get("to"); pc = (label_to_pc.get(str(tgt), pc+1) if not isinstance(tgt,int) else tgt)
                else:
                    pc += 1

            # ---- פונקציות/פריימים ----
            elif op == "FUNC":
                # דילוג על גוף בזמן קריאה — הקוד רץ רק כאשר CALLF מעביר pc לתוך הפונקציה
                # נתקדם עד ENDFUNC
                s = pc; e = None
                # נניח ש-prepass סימן e; אם לא — נמצא עכשיו:
                if ins.get("end") is not None:
                    e = int(ins["end"])
                else:
                    d = 1; i = pc+1
                    while i < N:
                        if program[i].get("op") == "FUNC": d += 1
                        if program[i].get("op") == "ENDFUNC":
                            d -= 1
                            if d==0: e=i; break
                        i += 1
                pc = (e+1) if e is not None else (pc+1)
            elif op == "ENDFUNC":
                # כשנכנסים לפה דרך RETURN/נפילה — נחזור דרך callstack
                if not callstack:
                    pc += 1
                else:
                    ret_pc, saved_regs = callstack.pop()
                    registers = saved_regs  # שחזור רגיסטרים של ההורה
                    ctx["__registers__"] = registers
                    pc = ret_pc
            elif op == "CALLF":
                fname = ins.get("name")
                if fname not in func_bounds:
                    raise VMError(f"no_such_function:{fname}")
                start, end = func_bounds[fname]
                # שמירת מסגרת
                callstack.append((pc+1, registers.copy()))
                # פריים חדש (רגיסטרים ריקים)
                registers = {}
                ctx["__registers__"] = registers
                # פרמטרים (אם יש) מתוך ins["args"]: [{"to":"reg:x","value":...}, ...]
                for a in ins.get("args", []):
                    registers[a["to"].split(":",1)[-1]] = resolve(a.get("value"))
                pc = start + 1  # אחרי FUNC
            elif op == "RET":
                if not callstack:
                    pc += 1
                else:
                    ret_pc, saved_regs = callstack.pop()
                    registers = saved_regs
                    ctx["__registers__"] = registers
                    pc = ret_pc

            # ---- Heap / אוספים מורכבים ----
            elif op == "NEW_OBJ":  # -> oid
                oid = next_oid; next_oid += 1; heap[oid] = {}
                push(oid); pc += 1
            elif op == "NEW_ARR":
                oid = next_oid; next_oid += 1; heap[oid] = []
                push(oid); pc += 1
            elif op == "SETK":  # SETK oid key value
                val = resolve(ins.get("value")); key = ins.get("key"); oid = resolve(ins.get("oid"))
                obj = heap.get(oid)
                if not isinstance(obj, dict): raise VMError("bad_obj_for_SETK")
                obj[key] = val; pc += 1
            elif op == "GETK":  # -> pushes obj[key]
                key = ins.get("key"); oid = resolve(ins.get("oid"))
                obj = heap.get(oid)
                if isinstance(obj, dict): push(obj.get(key))
                elif isinstance(obj, list) and isinstance(key,int) and 0<=key<len(obj): push(obj[key])
                else: push(None)
                pc += 1
            elif op == "APPEND":
                oid = resolve(ins.get("oid")); val = resolve(ins.get("value"))
                arr = heap.get(oid)
                if not isinstance(arr, list): raise VMError("bad_arr_for_APPEND")
                arr.append(val); pc += 1
            elif op == "LEN":
                oid = resolve(ins.get("oid"))
                obj = heap.get(oid)
                push(len(obj) if isinstance(obj,(list,dict)) else 0); pc += 1

            # ---- Timers / Sleep ----
            elif op == "SLEEP_MS":
                ms = int(resolve(ins.get("ms")) or 0)
                if ms < 0 or ms > self.limits.max_sleep_ms:
                    raise VMError("sleep_ms_limit")
                time.sleep(ms/1000.0); pc += 1

            # ---- IO סנדבוקס / Syscalls ----
            elif op == "FS_WRITE":
                io_calls += 1
                if io_calls > self.limits.io_calls_max: raise VMError("io_calls_exceeded")
                path = ins.get("path"); data = str(resolve(ins.get("data")) or "")
                FSSandbox.write_text(path, data)
                pc += 1
            elif op == "FS_READ":
                io_calls += 1
                if io_calls > self.limits.io_calls_max: raise VMError("io_calls_exceeded")
                path = ins.get("path"); data = FSSandbox.read_text(path)
                push(data); pc += 1
            elif op == "HTTP_GET":
                io_calls += 1
                if io_calls > self.limits.io_calls_max: raise VMError("io_calls_exceeded")
                url = ins.get("url")
                # שימוש בסנדבוקס רשת — נחזיר טקסט (או נזרוק חריגה)
                txt = NetSandbox.http_get_text(url, timeout=2.0)
                push(txt); pc += 1

            # ---- Evidence / Response ----
            elif op == "EVIDENCE":
                claim = ins.get("claim"); sources = list(ins.get("sources", []))
                if not claim: raise VMError("EVIDENCE_missing_claim")
                claims = ctx.setdefault("__claims__", [])
                claims.append({"claim": claim, "sources": sources})
                pc += 1
            elif op == "RESPOND":
                status = int(ins.get("status", 200))
                raw_body = ins.get("body", {})
                # החלפת reg:x לערך
                out = {}
                for k,v in raw_body.items(): out[k] = resolve(v)
                # FACT-GATE ENFORCEMENT
                try:
                    gate = FactGate()
                    gate.enforce(ctx, out, ctx.get("__fact_policy__"))
                    code, body = status, out
                except GroundValidationError as e:
                    code, body = 412, {"error":"precondition_failed","detail":str(e)}
                # סיום הפעלה
                break

            # ---- Async ----
            elif op == "SPAWN":
                if len(tasks) >= self.limits.max_async_tasks:
                    raise VMError("async_tasks_limit")
                tid = str(int(time.time()*1000)) + f"-{len(tasks)+1}"
                subprog = ins.get("body") or []
                subctx = {"__fact_policy__": ctx.get("__fact_policy__", {})}
                t = threading.Thread(target=_run_subtask, args=(tid, subprog, subctx), daemon=True)
                tasks[tid] = TaskResult(False, 202, {"spawned": True}, subctx, None)
                t.start()
                task_threads[tid] = t
                # expose task id
                registers[ins.get("as","task_id")] = tid
                pc += 1
            elif op == "JOIN":
                tid = resolve(ins.get("task"))
                tr = task_threads.get(tid)
                if tr: tr.join(timeout=ins.get("timeout_s", 5))
                # תוצאה
                tres = tasks.get(tid)
                if tres is None or not tres.ok:
                    push({"ok": False, "error": (tres.err if tres else "no_task")})
                else:
                    push({"ok": True, "code": tres.code, "body": tres.body})
                pc += 1

            # ---- Halt ----
            elif op == "HALT":
                break

            else:
                raise VMError(f"unknown_op:{op}")

            # בדיקת זיכרון
            if mem_kb_estimate() > self.limits.mem_kb_max:
                raise VMError("mem_kb_exceeded")

        # אם לא בוצעה RESPOND מפורשת — נחזיר 204
        return code, body, ctx