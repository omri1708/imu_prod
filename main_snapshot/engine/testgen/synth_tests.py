# engine/testgen/synth_tests.py
from __future__ import annotations
import os, json, textwrap
from typing import Any, Dict, List

TPL = """
# AUTO-GENERATED TESTS (basic smoke)
import json, os, re

def test_manifest_exists():
    assert os.path.exists({manifest!r})

def test_artifacts_list():
    arts = {arts_json}
    assert isinstance(arts, dict) and len(arts) >= 1
""".strip()


def synthesize_tests_for_artifacts(*, build_dir: str, manifest_path: str, artifacts: Dict[str,Any]) -> str:
    os.makedirs(os.path.join(build_dir, "tests"), exist_ok=True)
    fp = os.path.join(build_dir, "tests", "test_auto_smoke.py")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(TPL.format(manifest=manifest_path, arts_json=json.dumps(artifacts, ensure_ascii=False)))
    return fp