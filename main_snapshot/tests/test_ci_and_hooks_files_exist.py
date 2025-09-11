# tests/test_ci_and_hooks_files_exist.py
def test_workflows_and_policies_exist():
    assert open(".github/workflows/umbrella-e2e.yml","r",encoding="utf-8").read().startswith("name:")
    assert open("policy/rego/external_dns.rego","r",encoding="utf-8").read().startswith("package")
    assert open("policy/rego/ingress_tls.rego","r",encoding="utf-8").read().startswith("package")

def test_postsync_hook_job_exists():
    txt=open("helm/control-plane/templates/hooks/postsync-helmtest-rollback.yaml","r",encoding="utf-8").read()
    assert "helm test" in txt and "helm rollback" in txt