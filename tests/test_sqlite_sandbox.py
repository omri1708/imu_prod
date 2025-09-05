# tests/test_sqlite_sandbox.py
# -*- coding: utf-8 -*-
from adapters.db.sqlite_sandbox import SQLiteSandbox

def test_sqlite_roundtrip():
    s = SQLiteSandbox()
    s.migrate(["CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT)"])
    s.exec("INSERT INTO t(name) VALUES (?)", ("alice",))
    rows = s.query("SELECT * FROM t WHERE name=?", ("alice",))
    assert rows and rows[0]["name"] == "alice"