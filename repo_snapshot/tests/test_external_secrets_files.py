# tests/test_external_secrets_files.py

def test_external_secrets_files_exist():
    assert open("helm/umbrella/templates/external-secrets-store.yaml","r",encoding="utf-8").read().startswith("apiVersion:")
    assert open("helm/umbrella/templates/external-secrets.yaml","r",encoding="utf-8").read().startswith("apiVersion:")
