# imu_repo/ui_dsl/static_signer.py
from __future__ import annotations
import base64, hashlib
from typing import Tuple

def sri_sha256(b: bytes) -> str:
    h = hashlib.sha256(b).digest()
    return "sha256-" + base64.b64encode(h).decode("ascii")