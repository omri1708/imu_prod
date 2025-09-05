# contracts/errors.py
# -*- coding: utf-8 -*-
from typing import Optional

class ContractViolation(Exception):
    def __init__(self, msg: str, code: str = "contract_violation", detail: Optional[dict] = None):
        super().__init__(msg)
        self.code = code
        self.detail = detail or {}

class ResourceRequired(Exception):
    def __init__(self, resource: str, how_to_get: str):
        super().__init__(f"resource_required:{resource}")
        self.resource = resource
        self.how_to_get = how_to_get

class RateLimitExceeded(Exception):
    def __init__(self, scope: str, limit: str):
        super().__init__(f"rate_limit_exceeded:{scope}:{limit}")
        self.scope = scope
        self.limit = limit

class PolicyDenied(Exception):
    def __init__(self, reason: str, policy: dict):
        super().__init__(f"policy_denied:{reason}")
        self.reason = reason
        self.policy = policy

class SandboxDenied(Exception):
    def __init__(self, action: str, path_or_host: str):
        super().__init__(f"sandbox_denied:{action}:{path_or_host}")
        self.action = action
        self.path_or_host = path_or_host