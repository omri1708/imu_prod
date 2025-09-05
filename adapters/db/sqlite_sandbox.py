# adapters/db/sqlite_sandbox.py
# -*- coding: utf-8 -*-
import os, sqlite3, tempfile
from typing import List, Dict, Any

class SQLiteSandbox:
    def __init__(self, db_path: str = None):
        self._tmp = None
        self.db_path = db_path or self._mktemp()

    def _mktemp(self) -> str:
        td = tempfile.TemporaryDirectory()
        self._tmp = td
        return os.path.join(td.name, "db.sqlite")

    def migrate(self, stmts: List[str]):
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            for s in stmts: cur.execute(s)
            con.commit()

    def exec(self, sql: str, params: tuple = ()):
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor(); cur.execute(sql, params); con.commit()

    def query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor(); cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
