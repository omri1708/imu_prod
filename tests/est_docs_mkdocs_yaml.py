# tests/test_docs_mkdocs_yaml.py

def test_mkdocs_yaml_exists():
    assert open("mkdocs.yml","r",encoding="utf-8").read().startswith("site_name:")