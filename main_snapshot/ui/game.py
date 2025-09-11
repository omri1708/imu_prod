# imu_repo/ui/game.py
from __future__ import annotations
import os, subprocess
from typing import Dict, Any

class ResourceRequired(Exception): ...

class UnityAdapter:
    """Adapter for Unity projects: build, run, send commands."""
    def __init__(self,unity_path:str="/Applications/Unity/Hub/Editor"):
        if not os.path.exists(unity_path):
            raise ResourceRequired("Unity installation required")
        self.unity_path=unity_path

    def build(self,project_path:str,output_path:str,target:str="StandaloneOSX"):
        """Build a Unity project to specified target."""
        cmd=[self.unity_path,"-quit","-batchmode","-projectPath",project_path,
             "-buildTarget",target,"-executeMethod","BuildPipeline.BuildPlayer",
             "-logFile","-"]
        print(f"[Unity] Building {project_path} → {output_path}")
        subprocess.run(cmd,check=True)

    def run(self,exe_path:str):
        if not os.path.exists(exe_path):
            raise FileNotFoundError(exe_path)
        print(f"[Unity] Running {exe_path}")
        return subprocess.Popen([exe_path])

    def send_command(self,proc:subprocess.Popen,cmd:Dict[str,Any]):
        """Placeholder IPC: in practice use socket/pipe shared with Unity."""
        print(f"[Unity IPC] {cmd} → PID={proc.pid}")
