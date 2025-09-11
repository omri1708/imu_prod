# imu_repo/security/ed25519_optional.py
from __future__ import annotations
from typing import Optional, Tuple

_ED25519_OK = False
try:
    from nacl.signing import SigningKey, VerifyKey
    from nacl.encoding import HexEncoder
    _ED25519_OK = True
except Exception:
    _ED25519_OK = False

def ed25519_available() -> bool:
    return _ED25519_OK

def ed25519_keygen() -> Tuple[str, str]:
    """
    מחזיר (pub_hex, priv_hex). דורש pynacl; אם אין — יזרוק RuntimeError.
    """
    if not _ED25519_OK:
        raise RuntimeError("pynacl not available")
    sk = SigningKey.generate()
    vk = sk.verify_key
    return (vk.encode(encoder=HexEncoder).decode(), sk.encode(encoder=HexEncoder).decode())

def ed25519_sign(priv_hex: str, data: bytes) -> str:
    if not _ED25519_OK:
        raise RuntimeError("pynacl not available")
    sk = SigningKey(bytes.fromhex(priv_hex))
    sig = sk.sign(data).signature
    return sig.hex()

def ed25519_verify(pub_hex: str, data: bytes, sig_hex: str) -> bool:
    if not _ED25519_OK:
        raise RuntimeError("pynacl not available")
    vk = VerifyKey(bytes.fromhex(pub_hex))
    try:
        vk.verify(data, bytes.fromhex(sig_hex))
        return True
    except Exception:
        return False