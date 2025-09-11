# tests/test_dex_overlay_and_mermaid_gen.py

import os
def test_argocd_dex_overlay_exists():
    assert os.path.exists("argocd/overlays/dex/argocd-cm.yaml")
def test_mermaid_generator_script_exists():
    assert os.path.exists("scripts/gen_mermaid_from_values.py")