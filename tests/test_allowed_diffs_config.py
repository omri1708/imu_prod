# tests/test_allowed_diffs_config.py
import yaml
def test_allowed_diffs_yaml_valid():
    with open("scripts/allowed_diffs.yaml","r",encoding="utf-8") as f:
        y=yaml.safe_load(f)
    assert "kinds" in y and "resources" in y and "fields" in y
