# imu_repo/grounded/type_system.py
from __future__ import annotations
from typing import Optional

# קנוניקליזציה של טיפוסים לוגיים פשוטים
_CANON = {
    "str":"string", "string":"string", "text":"string", "varchar":"string",
    "int":"number", "integer":"number", "float":"number", "double":"number",
    "num":"number", "decimal":"number", "numeric":"number",
    "bool":"bool", "boolean":"bool",
    "date":"date",
    "datetime":"datetime", "timestamp":"datetime", "timestamptz":"datetime",
}

# התאמות־על אפשריות: date < datetime, number < string? (לא), string<->number? (לא)
def canon(t: Optional[str]) -> str:
    if not t: return "string"
    return _CANON.get(t.lower(), t.lower())

def is_compatible(ui_type: str, schema_type: str) -> bool:
    u, s = canon(ui_type), canon(schema_type)
    if u == s: return True
    # UI מבקש date, schema מספק datetime — מקובל (פיקוח על חיתוך זמן בשכבת הקליינט)
    if u == "date" and s == "datetime": return True
    return False