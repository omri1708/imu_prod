from __future__ import annotations
import json, glob
from pathlib import Path
import pytest

from engine.runtime_guard import check_runtime_table, RuntimeBlocked
from engine.auto_remediation import diagnose, propose_remedies, apply_remedies

def _fetcher_from_rows(rows):
    payload = json.dumps({"items": rows}).encode("utf-8")
    def f(url: str) -> bytes:
        assert url.startswith("https://api.example.com")
        return payload
    return f

@pytest.mark.parametrize("case_path", sorted(glob.glob("imu_repo/tests/generated/runtime_cases/*.json")))
def test_generated_runtime_cases(case_path: str, tmp_path: Path):
    case = json.loads(Path(case_path).read_text(encoding="utf-8"))
    spec   = case["spec"]
    policy = case["policy"]
    # צור state dir פר-ריצה כדי לשמור previous.json בנפרד
    if "runtime_state_dir" in policy:
        policy["runtime_state_dir"] = str(tmp_path / "rt_state")
    # שני מודלים: rows בלבד, או rows_v1/rows_v2 ל-drift
    if "rows" in case:
        rows = case["rows"]
        try:
            check_runtime_table(spec, policy=policy, fetcher=_fetcher_from_rows(rows))
        except RuntimeBlocked as rb:
            diags = diagnose(rb)
            rems  = propose_remedies(diags, policy=policy, table_specs=[spec])
            assert rems, f"no remedies for {case_path}"
            apply_remedies(rems, policy=policy, table_specs=[spec])
            out = check_runtime_table(spec, policy=policy, fetcher=_fetcher_from_rows(rows))
            assert out["ok"]
    else:
        # drift
        out1 = check_runtime_table(spec, policy=policy, fetcher=_fetcher_from_rows(case["rows_v1"]))
        assert out1["ok"]
        with pytest.raises(RuntimeBlocked) as blk:
            check_runtime_table(spec, policy=policy, fetcher=_fetcher_from_rows(case["rows_v2"]))
        diags = diagnose(blk.value)
        rems  = propose_remedies(diags, policy=policy, table_specs=[spec])
        assert rems
        apply_remedies(rems, policy=policy, table_specs=[spec])
        eff = dict(policy)
        eff["prev_content_hash"] = policy["runtime_prev_hash_map"][spec["binding_url"]]
        out2 = check_runtime_table(spec, policy=eff, fetcher=_fetcher_from_rows(case["rows_v2"]))
        assert out2["ok"]
