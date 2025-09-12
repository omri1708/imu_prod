import pathlib, re
BAD = re.compile(r"var[/\\]audit[/\\]pipeline(\.jsonl)?")
def test_no_hardcoded_pipeline_path():
    root = pathlib.Path(".")
    for p in root.rglob("**/*.py"):
        if "engine/telemetry/audit.py" in str(p):  # יוצא מן הכלל – שם יש ברירת מחדל
            continue
        txt = p.read_text("utf-8", "ignore")
        assert not BAD.search(txt), f"Hard-coded pipeline audit path in {p}"
