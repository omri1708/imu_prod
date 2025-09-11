# imu_repo/user_model/crypto_store.py
from __future__ import annotations
from typing import Dict, Any
import os, json, hmac, hashlib

def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    """מייצר זרם מפתחות ע"י HMAC(key, nonce||counter) בבלוקים של 32 בתים."""
    out = bytearray()
    ctr = 0
    while len(out) < length:
        msg = nonce + ctr.to_bytes(8, "big")
        block = hmac.new(key, msg, hashlib.sha256).digest()
        out.extend(block)
        ctr += 1
    return bytes(out[:length])

def encrypt_bytes(key: bytes, plaintext: bytes, *, nonce: bytes) -> bytes:
    ks = _keystream(key, nonce, len(plaintext))
    return bytes(a ^ b for a,b in zip(plaintext, ks))

def decrypt_bytes(key: bytes, ciphertext: bytes, *, nonce: bytes) -> bytes:
    # XOR סימטרי
    return encrypt_bytes(key, ciphertext, nonce=nonce)

def save_encrypted_json(path: str, key: bytes, obj: Dict[str,Any], *, nonce: bytes) -> None:
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ct = encrypt_bytes(key, b, nonce=nonce)
    with open(path,"wb") as f: f.write(ct)

def load_encrypted_json(path: str, key: bytes, *, nonce: bytes) -> Dict[str,Any]:
    with open(path,"rb") as f: ct = f.read()
    pt = decrypt_bytes(key, ct, nonce=nonce)
    return json.loads(pt.decode("utf-8"))