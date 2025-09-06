# server/policy/enforcement.py
from pydantic import BaseModel, Field
from typing import Literal, Optional
import re

class PolicyError(Exception):
    pass

class CapabilityRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=128)
    reason: Optional[str] = Field(None, max_length=4096)
    user_id: Optional[str] = None

class CapabilityPolicy:
    """
    Policy “לבקש ולהמשיך”:
    - אם capability זמינה – מחזירים 'already_available'.
    - אם מותר להתקין – 'install' (יתבצע ברקע).
    - אם אסור (לפי allow/deny lists, user-tier, OS, licensing) – PolicyError.
    """
    allow_patterns = [
        r"^android-sdk$", r"^ios-xcode$", r"^unity-cli$", r"^cuda-toolkit$",
        r"^k8s-cli$", r"^ffmpeg$", r"^webrtc$", r"^aiortc$", r"^libsrtp2?$",
    ]
    deny_patterns = [
        r"root-shell", r"unsigned-kernel-driver", r"unknown-binary",
    ]

    def decide(self, req: CapabilityRequest) -> Literal["already_available","install"]:
        from server.capabilities.registry import capability_registry
        cap = capability_registry.resolve(req.name)
        if not cap:
            raise PolicyError(f"unknown capability: {req.name}")

        # deny rules
        for pat in self.deny_patterns:
            if re.fullmatch(pat, req.name):
                raise PolicyError(f"capability denied by policy: {req.name}")

        # allow rules
        allowed = any(re.fullmatch(p, req.name) for p in self.allow_patterns)
        if not allowed:
            raise PolicyError(f"capability not allowed by policy: {req.name}")

        # availability check
        if cap.is_available():
            return "already_available"
        return "install"