import json
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from server.deps.evidence_gate import has_citations

async def evidence_enforcer(request: Request, call_next):
    resp = await call_next(request)

    # נאכף רק אם ה-dependency הציב דגל
    enforce = bool(getattr(request.state, "require_evidence", False))
    if not enforce:
        return resp

    # אל תיגע בזרמים/קבצים
    if isinstance(resp, StreamingResponse):
        return resp
    ctype = (resp.headers.get("content-type") or "").lower()
    if "application/json" not in ctype:
        return resp

    # קרא את הגוף (ייתכן iterator) ואז בנה מחדש את ה-Response
    body = b""
    if hasattr(resp, "body_iterator") and resp.body_iterator is not None:
        async for chunk in resp.body_iterator:
            body += chunk
    else:
        # JSONResponse לרוב כבר מחזיק body מוכן
        body = getattr(resp, "body", b"") or b""

    try:
        data = json.loads(body.decode("utf-8")) if body else {}
    except Exception:
        return JSONResponse({"error": "Missing/invalid JSON (evidence required)"}, status_code=422)

    if not has_citations(data):
        return JSONResponse({"error": "Missing evidence.citations (citations-or-silence)"}, status_code=422)

    return Response(
        content=body,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=getattr(resp, "media_type", "application/json"),
    )
