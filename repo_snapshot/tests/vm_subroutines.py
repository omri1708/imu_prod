# imu_repo/tests/vm_subroutines.py
from __future__ import annotations
from typing import List, Dict, Any
from core.vm.vm import VM, Limits

def program_sum() -> List[Dict[str,Any]]:
    return [
        {"op":"FUNC","name":"sum2"},
            {"op":"LOAD","reg":"a"},
            {"op":"LOAD","reg":"b"},
            {"op":"ADD"},
            {"op":"STORE","reg":"s"},
            {"op":"EVIDENCE","claim":"sum_result","sources":["bench:sum"]},
            {"op":"RESPOND","status":200,"body":{"sum":"reg:s"}},
        {"op":"ENDFUNC"},

        # main:
        {"op":"PUSH","value":21}, {"op":"STORE","reg":"a"},
        {"op":"PUSH","value":34}, {"op":"STORE","reg":"b"},
        {"op":"CALLF","name":"sum2","args":[{"to":"reg:a","value":"reg:a"},{"to":"reg:b","value":"reg:b"}]}
    ]

def program_structs() -> List[Dict[str,Any]]:
    return [
        {"op":"NEW_OBJ"}, {"op":"STORE","reg":"o"},
        {"op":"SETK","oid":"reg:o","key":"name","value":"IMU"},
        {"op":"NEW_ARR"}, {"op":"STORE","reg":"arr"},
        {"op":"APPEND","oid":"reg:arr","value":13},
        {"op":"APPEND","oid":"reg:arr","value":29},
        {"op":"SETK","oid":"reg:o","key":"vals","value":"reg:arr"},
        {"op":"GETK","oid":"reg:o","key":"vals"},
        {"op":"LEN","oid":"reg:o"},
        {"op":"EVIDENCE","claim":"fs_echo","sources":["bench:sum"]},
        {"op":"RESPOND","status":200,"body":{"size":"reg:LEN","obj":"reg:o"}} # הערה: LEN נשמר למחסנית; כאן רק מדגים
    ]

def run():
    vm = VM(Limits())
    c1,b1,_ = vm.run(program_sum(), {})
    print("sum:", c1, b1)
    c2,b2,_ = vm.run(program_structs(), {})
    print("structs:", c2, list(b2.keys()))
    return 0 if (c1==200 and c2==200 and "sum" in b1) else 1

if __name__=="__main__":
    raise SystemExit(run())