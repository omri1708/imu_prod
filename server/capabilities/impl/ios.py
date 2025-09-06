# server/capabilities/impl/ios.py
from typing import Tuple, Dict, Any
from server.capabilities.types import Capability

class IOSXcode(Capability):
    def name(self) -> str:
        return "ios-xcode"

    def is_available(self) -> bool:
        # macOS only; detect xcodebuild
        return self.which("xcodebuild")

    def install(self) -> Tuple[bool, Dict[str, Any]]:
        # Cannot programmatically install Xcode headless in a portable way.
        return False, {"hint": "Install Xcode from App Store or developer.apple.com; ensure xcodebuild exists"}