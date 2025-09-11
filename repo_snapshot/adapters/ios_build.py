# adapters/ios_build.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from contracts.base import ensure_tool, run_ok, Artifact, ResourceRequired
from provenance.store import ProvenanceStore

def _ensure_xcode():
    ensure_tool("xcodebuild", "Install Xcode (App Store) and ensure command line tools are installed: xcode-select --install")

def build_xcode(project_dir: str, scheme: str, configuration: str = "Release", sdk: str = "iphoneos", export_archive: bool = True, store: Optional[ProvenanceStore]=None) -> Artifact:
    """
    בונה IPA/ארכיון בעזרת xcodebuild. דורש macOS + Xcode מותקן.
    """
    if os.name != "posix":
        raise ResourceRequired("macOS with Xcode", "Run on a macOS host with Xcode.")
    _ensure_xcode()
    p = Path(project_dir).resolve()
    build_dir = p / "build"
    build_dir.mkdir(exist_ok=True)
    archive_path = build_dir / f"{scheme}.xcarchive"

    run_ok(["xcodebuild", "-scheme", scheme, "-configuration", configuration, "-sdk", sdk, "archive", "-archivePath", str(archive_path)], cwd=str(p))
    if export_archive:
        export_dir = build_dir / "export"
        export_dir.mkdir(exist_ok=True)
        # for simplicity, use automatic export; for real signing provide ExportOptions.plist
        run_ok(["xcodebuild", "-exportArchive", "-archivePath", str(archive_path), "-exportOptionsPlist", "ExportOptions.plist", "-exportPath", str(export_dir)], cwd=str(p))
        ipa = next(export_dir.rglob("*.ipa"), None)
        if not ipa:
            raise FileNotFoundError("no_ipa_found_after_export")
        art = Artifact(path=str(ipa), kind="ipa")
    else:
        art = Artifact(path=str(archive_path), kind="xcarchive")

    if store:
        art = store.add(art, trust_level="built-local", evidence={"builder": "xcodebuild"})
    return art