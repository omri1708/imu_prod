# provenance/castore.py
import hashlib, json, time
from typing import Optional

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def now_s() -> int:
    return int(time.time())

class ContentAddressableStore:
    """
    CAS מינימלי על דיסק (תיקייה אחת), מבוסס sha256. אין תלות חיצונית.
    """
    def __init__(self, root_dir: str):
        import os
        self.root = root_dir
        os.makedirs(self.root, exist_ok=True)

    def put(self, data: bytes) -> str:
        import os
        digest = sha256_bytes(data)
        path = os.path.join(self.root, digest)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(data)
        return digest

    def get(self, digest: str) -> Optional[bytes]:
        import os
        path = os.path.join(self.root, digest)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()