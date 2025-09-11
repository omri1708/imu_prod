# provenance/signing.py
# -*- coding: utf-8 -*-
"""
Content-addressable store + Ed25519 signing/verification.
"""
from __future__ import annotations
import os, hashlib, json, base64, time
from dataclasses import dataclass
from typing import Optional, Dict

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives import serialization
except Exception as e:
    raise RuntimeError("cryptography package required for signing") from e

STORE_DIR = ".imu/provenance/blobs"
META_DIR  = ".imu/provenance/meta"
KEY_DIR   = ".imu/keys"

os.makedirs(STORE_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)
os.makedirs(KEY_DIR, exist_ok=True)

def cas_put_bytes(data: bytes) -> str:
    h = hashlib.sha256(data).hexdigest()
    p = os.path.join(STORE_DIR, h)
    if not os.path.exists(p):
        with open(p,"wb") as f: f.write(data)
    return h

def cas_put_file(src_path: str) -> str:
    with open(src_path,"rb") as f: return cas_put_bytes(f.read())

def cas_get(hash_hex: str) -> bytes:
    p = os.path.join(STORE_DIR, hash_hex)
    with open(p,"rb") as f: return f.read()

def gen_keypair(name: str="default") -> Dict[str,str]:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()).decode()
    pub_pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    with open(os.path.join(KEY_DIR,f"{name}.priv.pem"),"w") as f: f.write(priv_pem)
    with open(os.path.join(KEY_DIR,f"{name}.pub.pem"),"w") as f: f.write(pub_pem)
    return {"private":priv_pem,"public":pub_pem}

def _load_priv(name:str="default")->Ed25519PrivateKey:
    from cryptography.hazmat.primitives import serialization
    pem = open(os.path.join(KEY_DIR,f"{name}.priv.pem"),"rb").read()
    return serialization.load_pem_private_key(pem,password=None)

def _load_pub(name:str="default")->Ed25519PublicKey:
    from cryptography.hazmat.primitives import serialization
    pem = open(os.path.join(KEY_DIR,f"{name}.pub.pem"),"rb").read()
    return serialization.load_pem_public_key(pem)

@dataclass
class Evidence:
    hash: str
    kind: str           # "http", "file", "calc", "sensor", "ui"
    url: Optional[str] = None
    meta: Optional[dict]= None
    ts: float = time.time()
    signer: Optional[str] = None
    sig_b64: Optional[str] = None
    trust: str = "unknown"  # low/medium/high/system

    def sign(self, key_name: str="default"):
        priv = _load_priv(key_name)
        payload = json.dumps({
            "hash": self.hash, "kind": self.kind, "url": self.url,
            "meta": self.meta, "ts": self.ts, "trust": self.trust
        }, sort_keys=True).encode()
        sig = priv.sign(payload)
        self.signer = key_name
        self.sig_b64 = base64.b64encode(sig).decode()
        # persist meta
        with open(os.path.join(META_DIR, f"{self.hash}.json"),"w",encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)

def verify_evidence(ev: Evidence) -> bool:
    pub = _load_pub(ev.signer or "default")
    payload = json.dumps({
        "hash": ev.hash, "kind": ev.kind, "url": ev.url,
        "meta": ev.meta, "ts": ev.ts, "trust": ev.trust
    }, sort_keys=True).encode()
    import base64
    sig = base64.b64decode(ev.sig_b64 or "")
    try:
        pub.verify(sig, payload)
        return True
    except Exception:
        return False