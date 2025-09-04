# imu_repo/ui/desktop.py
from __future__ import annotations
import tkinter as tk
from tkinter import scrolledtext, messagebox
import json
from typing import Callable, Dict, Any, List, Tuple

class DesktopError(Exception): ...

class DesktopApp:
    """
    Simple desktop UI using tkinter (standard library).
    - Binds to an Engine-like callable: run(program, payload) -> (status, body)
    """

    def __init__(self, run_fn: Callable[[List[Dict[str,Any]], Dict[str,Any]], Tuple[int,Dict[str,Any]]]):
        if not callable(run_fn):
            raise DesktopError("run_fn_must_be_callable")
        self.run_fn = run_fn
        self.root = tk.Tk()
        self.root.title("IMU Desktop")

        self.lbl_prog = tk.Label(self.root, text="Program (JSON list of ops):")
        self.lbl_prog.pack(anchor="w")
        self.txt_prog = scrolledtext.ScrolledText(self.root, height=12, width=100)
        self.txt_prog.pack(fill="both", expand=True)

        self.lbl_payload = tk.Label(self.root, text="Payload (JSON object):")
        self.lbl_payload.pack(anchor="w")
        self.txt_payload = scrolledtext.ScrolledText(self.root, height=6, width=100)
        self.txt_payload.pack(fill="both", expand=True)

        self.btn_run = tk.Button(self.root, text="Run", command=self._on_run)
        self.btn_run.pack(pady=6)

        self.lbl_out = tk.Label(self.root, text="Output:")
        self.lbl_out.pack(anchor="w")
        self.txt_out = scrolledtext.ScrolledText(self.root, height=12, width=100)
        self.txt_out.pack(fill="both", expand=True)

        # default program/payload
        default_prog = [
            {"op":"PUSH","ref":"$.payload.a"},
            {"op":"PUSH","ref":"$.payload.b"},
            {"op":"ADD"},
            {"op":"STORE","reg":"sum"},
            {"op":"EVIDENCE","claim":"a_plus_b_equals_sum","sources":["ui:desktop:default"]},
            {"op":"RESPOND","status":200,"body":{"sum":"reg:sum"}}
        ]
        self.txt_prog.insert("1.0", json.dumps(default_prog, ensure_ascii=False, indent=2))
        self.txt_payload.insert("1.0", json.dumps({"a":2,"b":3}, ensure_ascii=False, indent=2))

    def _on_run(self):
        try:
            program = json.loads(self.txt_prog.get("1.0", "end"))
            payload = json.loads(self.txt_payload.get("1.0", "end"))
            status, body = self.run_fn(program, payload)
            self.txt_out.delete("1.0", "end")
            self.txt_out.insert("1.0", f"STATUS: {status}\n{json.dumps(body, ensure_ascii=False, indent=2)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start(self):
        self.root.mainloop()
