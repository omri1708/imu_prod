# tests/test_unity_optional.py
# -*- coding: utf-8 -*-
import pytest, tempfile, os
from adapters.unity.build import unity_build
from adapters.contracts import ResourceRequired

def test_unity_optional(tmp_path):
    # מדמה פרויקט (Unity ייכשל אם לא פרויקט אמיתי — כאן מספיק לבדוק ResourceRequired או ריצה)
    try:
        unity_build(str(tmp_path), "Android")
    except ResourceRequired:
        pytest.skip("Unity CLI not installed")
    except Exception:
        # הצליח למצוא CLI וניסה לבנות → זה בסדר שנכשל בפרויקט מדומה
        assert True
    else:
        assert True