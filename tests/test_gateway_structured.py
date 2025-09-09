import json
import types
from engine.llm_gateway import LLMGateway

def test_structured_cache_and_cite_or_silence(monkeypatch):
    gw = LLMGateway()

    calls = {"n": 0}
    def fake_openai_chat_json(self, raw, schema_hint=None, temperature=0.0):
        calls["n"] += 1
        # מחזיר JSON עם _text ו-sources כדי לעבור cite-or-silence
        return json.dumps({"_text": "hello", "sources": ["https://example.com"]})
    # מוקים לפונקציה שה-Gateway שלך מצפה לה
    gw._openai_chat_json = types.MethodType(fake_openai_chat_json, gw)

    # ריצה ראשונה — צריכה לקרוא ל-"מודל"
    out1 = gw.structured(user_id="u", task="t", intent="i", schema_hint="{}", prompt="p",
                         content={"prompt": "p"}, temperature=0.0, require_grounding=False)
    assert out1["ok"] is True and out1["json"]["_text"] == "hello"
    assert calls["n"] == 1

    # ריצה שנייה עם אותו קונטקסט — צריכה להיות hit מה-cache (ללא עלייה במספר קריאות)
    out2 = gw.structured(user_id="u", task="t", intent="i", schema_hint="{}", prompt="p",
                         content={"prompt": "p"}, temperature=0.0, require_grounding=False)
    assert out2["ok"] is True
    assert calls["n"] == 1

def test_chat_cite_or_silence_block(monkeypatch):
    gw = LLMGateway()

    def fake_openai_chat(self, msgs, json_mode=False, temperature=0.0):
        # מחזיר טקסט ללא מקורות → אמור להיחסם
        return "statement with no sources"
    gw._openai_chat = types.MethodType(fake_openai_chat, gw)

    out = gw.chat(user_id="u", task="t", intent="i", content={"prompt":"p"}, require_grounding=False, temperature=0.0)
    assert out["ok"] is False and out["error"] == "no_citations"
