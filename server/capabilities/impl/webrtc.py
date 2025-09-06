# server/capabilities/impl/webrtc.py
from typing import Tuple, Dict, Any
from server.capabilities.types import Capability

class WebRTCBits(Capability):
    def name(self) -> str:
        return "webrtc"

    def is_available(self) -> bool:
        # Userspace tools for signaling/inspection; true runtime is in browsers or aiortc.
        return self.which("ffmpeg") or self.which("gst-launch-1.0")

    def install(self) -> Tuple[bool, Dict[str, Any]]:
        return False, {"hint": "Install ffmpeg or gstreamer for RTP tools; Python aiortc optional for server-side"}