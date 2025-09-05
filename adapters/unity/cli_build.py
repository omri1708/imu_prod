# adapters/unity/cli_build.py
# -*- coding: utf-8 -*-
import os
from ..contracts import ensure_tool, run, record_provenance

def build_unity(project_dir: str, build_target: str = "StandaloneWindows64", output_path: str = "Build/Game.exe") -> dict:
    """
    מריץ Unity במצב batchmode לבניית פרויקט.
    דורש התקנת Unity Editor תואם ו-Unity Hub/Editor על PATH.
    """
    # ניסיון: unity-editor או /Applications/Unity/Hub/Editor/*/Unity
    unity_bin = os.environ.get("UNITY_BIN") or "unity"
    ensure_tool(unity_bin, "Install Unity and set UNITY_BIN to editor binary")
    cmd = [
        unity_bin, "-batchmode", "-nographics",
        "-projectPath", project_dir,
        "-buildTarget", build_target,
        "-executeMethod", "BuildScript.PerformBuild",
        "-quit", "-logFile", "-"
    ]
    out = run(cmd, cwd=project_dir)
    if not os.path.exists(os.path.join(project_dir, output_path)):
        raise RuntimeError("Unity build output not found")
    prov = record_provenance("unity_build", {"project": project_dir, "target": build_target}, os.path.join(project_dir, output_path))
    return {"artifact": os.path.join(project_dir, output_path), "provenance": prov.__dict__, "log": out}

#TODO- הערה: דורש מחלקת C# בשם BuildScript בפרויקט שלך שמממשת PerformBuild,  (זה דפוס Unity מוכר).
#  אם אין — אפשר להחליף ל־-buildPlayer עם scene list.