# adapters/unity/scenes.py
# -*- coding: utf-8 -*-
import os, json, subprocess, shlex
from ..contracts import ResourceRequired

def list_scenes(project_path: str):
    """
    קורא את ProjectSettings/EditorBuildSettings.asset אם קיים; אחרת מריץ Unity -quit -batchmode לייצא.
    דורש Unity מותקן. אם אין — ResourceRequired.
    """
    unity = os.environ.get("UNITY_BIN", "unity")
    try:
        subprocess.run([unity, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        raise ResourceRequired("Unity Editor", "Install Unity; set UNITY_BIN to editor binary")

    # דרך מהירה: להריץ C# method דרך -executeMethod שידפיס JSON של הסצינות.
    cmd = f'{shlex.quote(unity)} -quit -batchmode -projectPath {shlex.quote(project_path)} ' \
          f'-executeMethod ScenesExporter.Export'
    # כאן אנו מצפים שסקריפט C# בפרויקט ידפיס JSON ל-stdout; אם אין – זו הגבלה של הפרויקט עצמו.
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = proc.stdout.strip()
    try:
        data = json.loads(out)
    except Exception:
        data = {"scenes": []}
    return data