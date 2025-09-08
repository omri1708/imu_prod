# -*- coding: utf-8 -*-
DOMAIN_FITNESS = {
  "entities": [
    {"name":"User", "fields":[["id","int"],["age","int"],["height_cm","int"],["weight_kg","float"],["level","str"]]},
    {"name":"WorkoutSession", "fields":[["id","int"],["user_id","int"],["date","str"],["minutes","int"],["intensity","int"],["type","str"]]},
    {"name":"Plan", "fields":[["id","int"],["user_id","int"],["start_date","str"],["weeks","int"],["goal","str"],["weekly_minutes","int"]]},
    {"name":"Recommendation", "fields":[["user_id","int"],["plan_id","int"],["next_session_type","str"],["target_minutes","int"],["rationale","str"]]}
  ],
  "events": [
    {"name":"logged_session","when":"user submits session log"},
    {"name":"missed_session","when":"session planned but not logged"}
  ],
  "rules": [
    "If adherence < 60% then reduce weekly_minutes by 10-20%",
    "If plateaus (no improvement 2 wks) then switch type (strength/cardio/hiit)",
    "If injury_risk high (intensity>8 and spikes>2/wk) then recommend deload week"
  ]
}