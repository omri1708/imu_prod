# imu_repo/engine/explore_policy.py
from __future__ import annotations

def decide_explore(history_len: int, epsilon: float) -> bool:
    """
    החלטה דטרמיניסטית: 'לחקור' כל N ריצות (N = round(1/epsilon)),
    כדי להמנע מרנדומליות (שימושי ל-CI).
    epsilon ∈ [0,1]; epsilon=0 → לעולם לא, epsilon>=1 → תמיד.
    """
    if epsilon <= 0.0:
        return False
    if epsilon >= 1.0:
        return True
    # כל N ריצות נחפש Explore
    n = max(1, round(1.0 / epsilon))
    # אם זו ריצה שמספרה מתחלק ב-n → Explore
    # (history_len הוא מספר ריצות שכבר בוצעו; הריצה הבאה היא history_len+1)
    return ((history_len + 1) % n) == 0