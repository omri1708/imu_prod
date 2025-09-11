# server/capabilities/registry.py
from typing import Dict, Optional
from server.capabilities.types import Capability
from server.capabilities.impl.android import AndroidSDK
from server.capabilities.impl.ios import IOSXcode
from server.capabilities.impl.unity import UnityCLI
from server.capabilities.impl.cuda import CUDAToolkit
from server.capabilities.impl.k8s import K8sCLI
from server.capabilities.impl.ffmpeg import FFmpeg
from server.capabilities.impl.webrtc import WebRTCBits

class CapabilityRegistry:
    def __init__(self):
        self._caps: Dict[str, Capability] = {
            "android-sdk": AndroidSDK(),
            "ios-xcode": IOSXcode(),
            "unity-cli": UnityCLI(),
            "cuda-toolkit": CUDAToolkit(),
            "k8s-cli": K8sCLI(),
            "ffmpeg": FFmpeg(),
            "webrtc": WebRTCBits(),
            # libsrtp can be bundled as part of webrtc or separate:
            "libsrtp2": WebRTCBits(),  # logical alias
        }

    def resolve(self, name: str) -> Optional[Capability]:
        return self._caps.get(name)

capability_registry = CapabilityRegistry()