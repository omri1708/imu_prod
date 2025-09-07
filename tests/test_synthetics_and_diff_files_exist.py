# tests/test_synthetics_and_diff_files_exist.py
def test_ci_diff_workflow_exists():
    txt=open(".github/workflows/umbrella-diff.yml","r",encoding="utf-8").read()
    assert "umbrella-diff" in txt

def test_diff_script_runs_imports():
    import importlib.util, sys
    spec=importlib.util.spec_from_file_location("diff_umbrella","scripts/diff_umbrella.py")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert hasattr(mod,"load_docs") and hasattr(mod,"main")

def test_k6_config_and_hook_exist():
    assert open("helm/control-plane/templates/k6-configmap.yaml","r",encoding="utf-8").read().startswith("apiVersion:")
    assert open("helm/control-plane/templates/hooks/postsync-synthetics-rollback.yaml","r",encoding="utf-8").read().startswith("apiVersion:")