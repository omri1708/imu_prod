# server/capabilities/types.py
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import shutil
import subprocess

class Capability(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def install(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Returns (ok, meta) and should not raise.
        Implementations must be safe (no interactive prompts).
        """
        ...

    def which(self, bin_name: str) -> bool:
        return shutil.which(bin_name) is not None

    def run(self, cmd: list[str]) -> Tuple[bool, str]:
        try:
            out = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True, out.stdout.strip()
        except subprocess.CalledProcessError as e:
            return False, (e.stdout or "") + (e.stderr or "")