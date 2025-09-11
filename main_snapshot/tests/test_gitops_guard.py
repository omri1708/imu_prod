# tests/test_gitops_guard.py
from __future__ import annotations
from fastapi.testclient import TestClient
from server.http_api import APP
from gitops.utils import have_git

client = TestClient(APP)

def test_git_verify_signatures_grace(tmp_path):
    # יוצר ריפו דמה בלי חתימות — הפונקציה עשויה לכשל אבל חייבת להשיב JSON
    repo = tmp_path/"r"; repo.mkdir()
    # נריץ init דרך gitops api כדי למנוע תלות ישירה ב-git כאן
    r = client.post("/gitops/init", json={"user_id":"demo-user","path":str(repo)})
    assert r.status_code == 200
    r = client.post("/gitops/guard/git/verify_signatures", json={"repo_path":str(repo), "rev_range":"HEAD..HEAD"})
    assert r.status_code == 200
    j = r.json()
    # או resource_required=git (אם git לא זמין), או תשובה תקפה
    assert ("resource_required" in j) or ("good" in j)