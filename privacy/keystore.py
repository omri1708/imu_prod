# imu_repo/privacy/keystore.py
from __future__ import annotations
import os, json, hmac, hashlib, base64, struct
from typing import Tuple

BASE = "/mnt/data/imu_repo/.users"
KEYS = os.path.join(BASE, ".keys")
os.makedirs(KEYS, exist_ok=True)

def _key_path(user_id: str) -> str:
    return os.path.join(KEYS, f"{user_id}.key")

def get_or_create_key(user_id: str) -> bytes:
    p = _key_path(user_id)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return f.read()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    key = os.urandom(32)  # 256-bit
    with open(p, "wb") as f:
        f.write(key)
    return key

def _keystream_block(key: bytes, nonce: bytes, counter: int) -> bytes:
    # SHA256(key || nonce || counter_le)
    m = hashlib.sha256()
    m.update(key)
    m.update(nonce)
    m.update(struct.pack("<Q", counter))
    return m.digest()

def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for (x, y) in zip(a, b))

def encrypt_bytes(user_id: str, plain: bytes) -> dict:
    key = get_or_create_key(user_id)
    nonce = os.urandom(16)
    out = bytearray()
    counter = 0
    for i in range(0, len(plain), 32):
        block = plain[i:i+32]
        ks = _keystream_block(key, nonce, counter)
        out.extend(_xor_bytes(block, ks[:len(block)]))
        counter += 1
    ct = bytes(out)
    mac = hmac.new(key, nonce + ct, hashlib.sha256).digest()
    return {
        "v": 1,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ct": base64.b64encode(ct).decode("ascii"),
        "mac": base64.b64encode(mac).decode("ascii"),
    }

def decrypt_bytes(user_id: str, obj: dict) -> bytes:
    key = get_or_create_key(user_id)
    nonce = base64.b64decode(obj["nonce"])
    ct = base64.b64decode(obj["ct"])
    mac = base64.b64decode(obj["mac"])
    mac2 = hmac.new(key, nonce + ct, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, mac2):
        raise ValueError("bad_mac")
    out = bytearray()
    counter = 0
    for i in range(0, len(ct), 32):
        block = ct[i:i+32]
        ks = _keystream_block(key, nonce, counter)
        out.extend(_xor_bytes(block, ks[:len(block)]))
        counter += 1
    return bytes(out)