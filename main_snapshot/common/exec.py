# common/exc.py
# -*- coding: utf-8 -*-
from typing import List

class ResourceRequired(Exception):
    """נזרק כשיש יכולת שנדרשת מנוע/SDK/הרשאה חיצונית – לא ממציאים, אלא מבקשים מפורשות."""
    def __init__(self, kind: str, items: List[str], how_to: str):
        super().__init__(f"{kind} required: {items} -> {how_to}")
        self.kind = kind
        self.items = items
        self.how_to = how_to