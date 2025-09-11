# imu_repo/user_model/crypto_utils.py
from __future__ import annotations
from typing import Tuple
import os, hmac, hashlib, base64, json

def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """HKDF-Extract+Expand (RFC5869, גרסה מינימלית)."""
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    t = b""; okm = b""; i = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t; i += 1
    return okm[:length]

def derive_keys(master: bytes) -> Tuple[bytes, bytes]:
    """נגזר שני מפתחות: הצפנה ו־MAC."""
    enc = hkdf_sha256(master, b"imu.salt", b"enc", 32)
    mac = hkdf_sha256(master, b"imu.salt", b"mac", 32)
    return enc, mac

def _keystream(enc_key: bytes, nonce: bytes, nbytes: int) -> bytes:
    """מחולל זרם מפתחות דרך HMAC(key, nonce||counter)."""
    out = b""; c = 0
    while len(out) < nbytes:
        block = hmac.new(enc_key, nonce + c.to_bytes(8, "big"), hashlib.sha256).digest()
        out += block; c += 1
    return out[:nbytes]

def seal(plaintext: bytes, master: bytes) -> str:
    """מצפין ומאמת: מחזיר JSON קומפקטי b64 (nonce, ct, tag)."""
    enc_key, mac_key = derive_keys(master)
    nonce = os.urandom(16)
    ks = _keystream(enc_key, nonce, len(plaintext))
    ct = bytes(a ^ b for a, b in zip(plaintext, ks))
    tag = hmac.new(mac_key, nonce + ct, hashlib.sha256).digest()
    obj = {
        "n": base64.b64encode(nonce).decode(),
        "c": base64.b64encode(ct).decode(),
        "t": base64.b64encode(tag).decode()
    }
    return json.dumps(obj, separators=(",",":"))

def open_sealed(payload: str, master: bytes) -> bytes:
    """מאמת ומפענח JSON שהוחזר מ-seal(). זורק ValueError אם נכשל."""
    enc_key, mac_key = derive_keys(master)
    obj = json.loads(payload)
    nonce = base64.b64decode(obj["n"])
    ct    = base64.b64decode(obj["c"])
    tag   = base64.b64decode(obj["t"])
    exp = hmac.new(mac_key, nonce + ct, hashlib.sha256).digest()
    if not hmac.compare_digest(exp, tag):
        raise ValueError("bad_tag")
    ks = _keystream(enc_key, nonce, len(ct))
    pt = bytes(a ^ b for a, b in zip(ct, ks))
    return pt