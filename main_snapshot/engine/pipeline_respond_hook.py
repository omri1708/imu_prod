# imu_repo/engine/pipeline_respond_hook.py
from __future__ import annotations
from typing import Any, Dict
from engine.agent_emit import agent_emit_answer

def pipeline_respond(*, ctx: Dict[str,Any], answer_text: str) -> Dict[str,Any]:
    """
    Hook אחיד להפקת תשובה עם אכיפת הוכחות.
    אם ב-ctx יש __http_fetcher__ — הוא ייאסף ע"י הגשר אוטומטית.
    """
    policy = ctx.get("__policy__", {}) if isinstance(ctx, dict) else {}
    return agent_emit_answer(answer_text=answer_text, ctx=ctx, policy=policy)

#TODO- הערה: בבדיקה האחרונה הראיתי שברירת המחדל ללא רשת תחסום http evidence —
#  זה נכון ורצוי בסביבות CI;
#  המערכת תעבור עם evidence מסוג inline או כשהטרנספורט HTTP מוזרק 
# (אם תרצה, נוכל להוסיף פרמטר fetcher ב־pipeline_respond שיגיע מה־ctx).