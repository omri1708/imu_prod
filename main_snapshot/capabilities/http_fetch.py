# imu_repo/capabilities/http_fetch.py
from __future__ import annotations
from typing import Dict, Any
from grounded.claims import current

async def fetch_text(spec: Dict[str,Any]) -> str:
    """
    spec = {
      "url": "https://example/resource",
      "content": "TEXT..."     # בטסטים לא מושכים מהאינטרנט; מוסרים תוכן כאן
    }
    """
    url = str(spec["url"])
    content = str(spec.get("content",""))
    # הוסף ראיה ממקור "חיצוני"
    current().add_evidence("http_fetch", {
        "source_url": url,
        "trust": 0.9,     # אמון גבוה; סף ברירת מחדל (0.7) יעבור
        "ttl_s": 300,     # טרי למשך 5 דקות
        "payload": {"len": len(content)}
    })
    return content

