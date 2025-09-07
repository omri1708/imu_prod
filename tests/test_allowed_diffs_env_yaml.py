# tests/test_allowed_diffs_env_yaml.py
import yaml
def test_allowed_diffs_has_envs():
    y=yaml.safe_load(open("scripts/allowed_diffs.yaml","r",encoding="utf-8"))
    assert "environments" in y and "dev" in y["environments"]
