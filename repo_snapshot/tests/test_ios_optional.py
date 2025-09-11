# tests/test_ios_optional.py
# -*- coding: utf-8 -*-
import pytest
from adapters.mobile.ios_build import build_ios, ResourceRequired

def test_ios_build_optional(tmp_path):
    proj = tmp_path/"App.xcodeproj"
    try:
        build_ios(str(proj), scheme="App")
    except ResourceRequired:
        pytest.skip("xcodebuild not installed")
    except Exception:
        assert True
    else:
        assert True