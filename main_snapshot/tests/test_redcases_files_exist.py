# tests/test_redcases_files_exist.py
def test_redcases_values_files():
    for p in (
        "tests/redcases/values.bad-ingress-no-tls.yaml",
        "tests/redcases/values.bad-ingress-disallowed-class.yaml",
        "tests/redcases/values.bad-externaldns-off.yaml",
        "tests/redcases/values.bad-certmanager-no-email.yaml",
    ):
        assert open(p,"r",encoding="utf-8").read().strip() != ""