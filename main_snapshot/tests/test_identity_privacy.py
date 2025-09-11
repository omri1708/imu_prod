# tests/test_identity_privacy.py
# -*- coding: utf-8 -*-
import os, pytest
from identity.profile_store import ProfileStore, Profile

def test_profile_save_load_roundtrip(tmp_path):
    root = tmp_path / "id"
    keys = tmp_path / "keys" / "k.key"
    ps = ProfileStore(str(root), str(keys))
    p = Profile(user_id="u1", traits={"lang":"he"}, goals={"ship":0.9}, culture={}, affect={"valence":0.1})
    ps.save(p)
    p2 = ps.load("u1")
    assert p2 and p2.user_id == "u1"

def test_ttl_expire(tmp_path, monkeypatch):
    root = tmp_path / "id"
    keys = tmp_path / "keys" / "k.key"
    ps = ProfileStore(str(root), str(keys))
    p = Profile(user_id="u2", traits={}, goals={}, culture={}, affect={}, ttl_sec=0)
    ps.save(p)
    # טעינה מיידית אמורה למחוק
    out = ps.load("u2")
    assert out is None