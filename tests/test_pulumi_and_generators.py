# tests/test_pulumi_and_generators.py
import os, json
def test_pulumi_alerts_project_files():
    assert os.path.exists("infra/pulumi/alerts-app-patch/index.ts")
    assert os.path.exists("infra/pulumi/alerts-app-patch/Pulumi.yaml")
    assert os.path.exists("infra/pulumi/alerts-app-patch/package.json")

def test_gen_alerts_values_script():
    p="scripts/gen_alerts_values.sh"
    assert os.path.exists(p)
    assert os.access(p, os.X_OK)