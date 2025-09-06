# server/capabilities/impl/unity.py
from typing import Tuple, Dict, Any
from server.capabilities.types import Capability

class UnityCLI(Capability):
    def name(self) -> str:
        return "unity-cli"

    def is_available(self) -> bool:
        # Unity Hub or unity CLI available?
        return self.which("unity") or self.which("Unity")

    def install(self) -> Tuple[bool, Dict[str, Any]]:
        # Unity Hub typically interactive/EULA; point user
        return False, {"hint": "Install Unity Hub/Editor; ensure 'unity' CLI on PATH"}