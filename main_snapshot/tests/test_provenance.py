# tests/test_provenance.py
# -*- coding: utf-8 -*-
import os, pytest
from provenance.store import CASStore

def test_cas_put_and_verify(tmp_path):
    cas = CASStore(str(tmp_path/"cas"), str(tmp_path/"keys"))
    meta = cas.put_bytes(b"hello", sign=True, url="https://example.org/", trust=0.9, not_after_days=1)
    assert cas.verify_meta(meta) is True