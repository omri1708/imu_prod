# server/capabilities/impl/android.py
from typing import Tuple, Dict, Any
from server.capabilities.types import Capability
import os

class AndroidSDK(Capability):
    def name(self) -> str:
        return "android-sdk"

    def is_available(self) -> bool:
        # Heuristics: sdkmanager or gradle + ANDROID_HOME
        return self.which("sdkmanager") or (self.which("gradle") and ("ANDROID_HOME" in os.environ))

    def install(self) -> Tuple[bool, Dict[str, Any]]:
        # Best-effort, non-interactive. In real machines this would install cmdline-tools.
        meta = {"attempts":[]}
        # Try gradle first (often enough for building with preconfigured sdk)
        ok = False
        if not self.which("gradle"):
            meta["attempts"].append("gradle: not found")
        else:
            meta["attempts"].append("gradle: present")
            ok = True
        # We don’t curl | sh here for safety; report requirement if missing.
        if not ok:
            meta["hint"] = "Install Android cmdline-tools and accept licenses"
        return ok, meta