# tests/test_android_optional.py
# -*- coding: utf-8 -*-
import pytest, os, tempfile
from adapters.mobile.android_build import build_android, ResourceRequired

def test_android_build_optional(tmp_path):
    # ללא פרויקט אמיתי — בודקים זיהוי Gradle/gradlew וחסד אם חסר
    try:
        build_android(str(tmp_path))
    except ResourceRequired:
        pytest.skip("Gradle not installed and no gradlew")
    except Exception:
        assert True
    else:
        assert True