# common/errors.py
# -*- coding: utf-8 -*-
class ContractError(Exception):
    pass

class EvidenceMissing(Exception):
    pass

class ResourceRequired(Exception):
    """
    נזרקת כשצריך מנוע/SDK/כלי חיצוני כדי לבצע בפועל.
    message: תיאור אנושי קצר
    need: dict עם פירוט מדויק מה חסר ואיך מספקים
    """
    def __init__(self, message: str, need: dict):
        super().__init__(message)
        self.need = need