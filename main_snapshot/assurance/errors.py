# -*- coding: utf-8 -*-
from __future__ import annotations

class AssuranceError(Exception):
    pass

class RefusedNotGrounded(AssuranceError):
    """Raised if claims/evidence/validators do not satisfy the Gate."""
    def __init__(self, reason: str):
        super().__init__(f"refused_not_grounded: {reason}")
        self.reason = reason

class ResourceRequired(AssuranceError):
    """Raised when a required external resource/tool/credential is missing."""
    def __init__(self, what: str, how_to_get: str = ""):
        msg = f"resource_required: {what}"
        if how_to_get:
            msg += f" | obtain: {how_to_get}"
        super().__init__(msg)
        self.what = what
        self.how_to_get = how_to_get

class ValidationFailed(AssuranceError):
    def __init__(self, details: str):
        super().__init__(f"validation_failed: {details}")
        self.details = details
