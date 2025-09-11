# imu_repo/user_model/emotion.py
from __future__ import annotations
from typing import Dict

_POS = {"great","happy","love","excellent","awesome","fantastic","טוב","מצוין","נהדר"}
_NEG = {"bad","sad","hate","terrible","awful","angry","גרוע","נורא","כועס"}
_FEAR= {"scared","afraid","worried","דואג","מפחד","לחוץ"}
_JOY = {"joy","glad","smile","שמחה","שמח"}
_ANGER={"mad","furious","rage","כועס","זועם"}
# TODO- לא קשיחים, חיבור ל NLU/UI
def detect(text: str) -> Dict[str,float]:
    t = set((text or "").lower().split())
    def score(words): 
        return 1.0 if any(w in t for w in words) else 0.0
    pos = score(_POS); neg = score(_NEG); fear=score(_FEAR); joy=score(_JOY); anger=score(_ANGER)
    total = pos+neg+fear+joy+anger
    if total==0: return {"neutral":1.0}
    return {
        "positive": pos/total,
        "negative": neg/total,
        "fear": fear/total,
        "joy": joy/total,
        "anger": anger/total
    }