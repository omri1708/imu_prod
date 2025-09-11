# imu_repo/tests/benchmarks.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any

def _prog_sum() -> List[Dict[str,Any]]:
    return [
        {"op":"PUSH","ref":"$.payload.x"},
        {"op":"PUSH","ref":"$.payload.y"},
        {"op":"ADD"},
        {"op":"STORE","reg":"s"},
        {"op":"EVIDENCE","claim":"sum_ok","sources":["bench:sum"]},
        {"op":"RESPOND","status":200,"body":{"sum":"reg:s"}}
    ]

def _prog_loop(n:int=50) -> List[Dict[str,Any]]:
    # לולאה ע"י decrement במדדים (מימוש VM שלך תומך ב-JUMP/JZ/JNZ)
    prog=[
        {"op":"PUSH","value":0},
        {"op":"STORE","reg":"acc"},
        {"op":"PUSH","value":n},
        {"op":"STORE","reg":"i"},
        {"op":"LABEL","name":"L0"},
        {"op":"LOAD","reg":"i"},
        {"op":"PUSH","value":0},
        {"op":"JZ","label":"END"},
        {"op":"LOAD","reg":"acc"},
        {"op":"LOAD","reg":"i"},
        {"op":"ADD"},
        {"op":"STORE","reg":"acc"},
        {"op":"LOAD","reg":"i"},
        {"op":"PUSH","value":1},
        {"op":"SUB"},
        {"op":"STORE","reg":"i"},
        {"op":"JUMP","label":"L0"},
        {"op":"LABEL","name":"END"},
        {"op":"EVIDENCE","claim":"loop_ok","sources":["bench:loop"]},
        {"op":"RESPOND","status":200,"body":{"acc":"reg:acc"}}
    ]
    return prog

def _prog_io() -> List[Dict[str,Any]]:
    # IO קל: כתיבה/קריאה בסנדבוקס
    return [
        {"op":"PUSH","value":"hello"},
        {"op":"STORE","reg":"msg"},
        {"op":"CAP","name":"fs_write","args":{"rel":"bench/hello.txt","content":"reg:msg"}},
        {"op":"CAP","name":"fs_read","args":{"rel":"bench/hello.txt"}},
        {"op":"STORE","reg":"read"},
        {"op":"EVIDENCE","claim":"io_ok","sources":["bench:fs"]},
        {"op":"RESPOND","status":200,"body":{"echo":"reg:read"}}
    ]

def default_suite() -> List[Tuple[List[Dict[str,Any]], Dict[str,Any]]]:
    return [
        (_prog_sum(), {"x": 13, "y": 29}),
        (_prog_loop(100), {}),
        (_prog_io(), {})
    ]
