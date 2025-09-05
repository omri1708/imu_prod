# adapters/unity_cli.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from contracts.base import ensure_tool, run_ok, Artifact, ResourceRequired
from provenance.store import ProvenanceStore

def _unity_path() -> str:
    # תרצה להציב UNITY_PATH=/Applications/Unity/Hub/Editor/2022.3.XXf1/Unity.app/Contents/MacOS/Unity (וכו')
    path = os.environ.get("UNITY_PATH")
    if not path:
        raise ResourceRequired("UNITY_PATH", "Set UNITY_PATH to your Unity editor binary (batchmode-capable).")
    return path

def build_unity_project(project_dir: str, build_target: str = "StandaloneLinux64", output_path: Optional[str] = None, store: Optional[ProvenanceStore]=None) -> Artifact:
    """
    מריץ Unity בבאטצ'ומוד לבנות חבילה.
    build_target דוגמאות: StandaloneWindows64 / StandaloneOSX / StandaloneLinux64 / Android / iOS
    """
    unity = _unity_path()
    p = Path(project_dir).resolve()
    out = Path(output_path or (p / "Build" / f"build-{build_target}"))
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        unity, "-quit", "-batchmode",
        "-projectPath", str(p),
        "-buildTarget", build_target,
        "-executeMethod", "BuildScript.Build",  # מצופה סקריפט C# בפרויקט
        "-logFile", str(p / "unity_build.log"),
        "-buildOutput", str(out)  # custom arg לצדו של BuildScript שלך
    ]
    run_ok(cmd)
    # נאסוף ארטיפקט: תיקיית Build או קובץ בודד
    art_path = out if out.exists() else (p / "Build")
    if not art_path.exists():
        raise FileNotFoundError("unity_build_output_missing")
    kind = "unity-bundle"
    art = Artifact(path=str(art_path), kind=kind)
    if store:
        art = store.add(art, trust_level="built-local", evidence={"builder": "unity-batch"})
    return art