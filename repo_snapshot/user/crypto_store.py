# imu_repo/user/crypto_store.py
from __future__ import annotations
import os, json, hmac, hashlib, struct, base64
from typing import Any, Dict, Optional

class ResourceRequired(Exception): ...

# Backend 1: cryptography (עדיף)
try:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False

def _pbkdf2_key(password: str, salt: bytes, length: int = 32) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000, dklen=length)

class _XORDrbg:
    """Keystream גנרי על בסיס HMAC-SHA256 (לא קריפטוגרפי חזק)."""
    def __init__(self, key: bytes):
        self.key = key
        self.counter = 0

    def stream(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            block = hmac.new(self.key, struct.pack(">Q", self.counter), hashlib.sha256).digest()
            self.counter += 1
            out.extend(block)
        return bytes(out[:n])

def _xor_encrypt(key: bytes, plaintext: bytes) -> bytes:
    drbg = _XORDrbg(key)
    ks = drbg.stream(len(plaintext))
    return bytes(a ^ b for a, b in zip(plaintext, ks))

class EncryptedJSONStore:
    """
    אחסון JSON מוצפן בקובץ יחיד.
    - אם HAS_CRYPTO: AES-GCM עם Scrypt KDF.
    - אחרת: XOR-DRBG; אם strict_security=True → נזרקת ResourceRequired.
    """

    def __init__(self, path: str, password: str, strict_security: bool = False):
        self.path = path
        self.password = password
        self.strict = strict_security
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._write_blob({})

    def _derive_key(self, salt: bytes) -> bytes:
        if HAS_CRYPTO:
            kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
            return kdf.derive(self.password.encode("utf-8"))
        return _pbkdf2_key(self.password, salt, 32)

    def _read_blob(self) -> Dict[str, Any]:
        with open(self.path, "rb") as f:
            blob = f.read()
        if not blob:
            return {}
        salt, payload = blob[:16], blob[16:]
        key = self._derive_key(salt)
        if HAS_CRYPTO:
            nonce, ct = payload[:12], payload[12:]
            aes = AESGCM(key)
            data = aes.decrypt(nonce, ct, None)
        else:
            if self.strict:
                raise ResourceRequired("cryptography", "pip install cryptography")
            data = _xor_encrypt(key, payload)
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return {}

    def _write_blob(self, obj: Dict[str, Any]) -> None:
        salt = os.urandom(16)
        key = self._derive_key(salt)
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        if HAS_CRYPTO:
            aes = AESGCM(key)
            nonce = os.urandom(12)
            ct = aes.encrypt(nonce, data, None)
            payload = nonce + ct
        else:
            if self.strict:
                raise ResourceRequired("cryptography", "pip install cryptography")
            payload = _xor_encrypt(key, data)
        with open(self.path, "wb") as f:
            f.write(salt + payload)

    def get(self, key: str, default: Any = None) -> Any:
        return self._read_blob().get(key, default)

    def put(self, key: str, value: Any) -> None:
        obj = self._read_blob()
        obj[key] = value
        self._write_blob(obj)

    def delete(self, key: str) -> None:
        obj = self._read_blob()
        if key in obj:
            del obj[key]
            self._write_blob(obj)

    def all(self) -> Dict[str, Any]:
        return self._read_blob()



class CryptoStore:
    def __init__(self, path: str = ".imu_state/crypto.key"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            key = os.urandom(32)
            with open(self.path, "wb") as f: f.write(key)
        with open(self.path, "rb") as f: self.key = f.read()

    def sign(self, data: bytes) -> str:
        mac = hmac.new(self.key, data, hashlib.sha256).digest()
        return base64.b64encode(mac).decode()

    def verify(self, data: bytes, sig_b64: str) -> bool:
        try:
            mac = base64.b64decode(sig_b64.encode())
            exp = hmac.new(self.key, data, hashlib.sha256).digest()
            return hmac.compare_digest(mac, exp)
        except Exception:
            return False