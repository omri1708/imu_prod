# imu_repo/engine/pipeline_respond_hook.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from engine.agent_emit import agent_emit_answer

def pipeline_respond(*, ctx: Dict[str,Any], answer_text: str) -> Dict[str,Any]:
    """
    Hook פשוט שניתן לקרוא אליו מכל מקום בפייפליין כדי להפיק תשובה.
    הוא ייחסם אם אין ראיות כנדרש במדיניות.
    """
    policy = ctx.get("__policy__", {}) if isinstance(ctx, dict) else {}
    return agent_emit_answer(answer_text=answer_text, ctx=ctx, policy=policy)

#TODO- הערה: בבדיקה האחרונה הראיתי שברירת המחדל ללא רשת תחסום http evidence —
#  זה נכון ורצוי בסביבות CI;
#  המערכת תעבור עם evidence מסוג inline או כשהטרנספורט HTTP מוזרק 
# (אם תרצה, נוכל להוסיף פרמטר fetcher ב־pipeline_respond שיגיע מה־ctx).