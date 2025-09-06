# server/capabilities/impl/ffmpeg.py
from typing import Tuple, Dict, Any
from server.capabilities.types import Capability

class FFmpeg(Capability):
    def name(self) -> str:
        return "ffmpeg"

    def is_available(self) -> bool:
        return self.which("ffmpeg")

    def install(self) -> Tuple[bool, Dict[str, Any]]:
        return False, {"hint": "Install ffmpeg via your OS package manager"}