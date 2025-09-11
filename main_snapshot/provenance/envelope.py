# provenance/envelope.py
# DSSE-like envelope for CAS digests, signed with Ed25519.
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any
import json, base64, time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

PAYLOAD_TYPE = "application/vnd.imu.cas-record+json"

@dataclass
class Signature:
    kid: str
    alg: str
    sig_b64: str

@dataclass
class Envelope:
    payloadType: str
    payload_b64: str
    signatures: list[Signature]

def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

def _ub64(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))

def _canonical(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",",":")).encode("utf-8")

def sign_cas_record(priv: Ed25519PrivateKey, kid: str, record: Dict[str, Any]) -> Envelope:
    payload = _canonical(record)
    sig = priv.sign(payload)
    env = Envelope(
        payloadType=PAYLOAD_TYPE,
        payload_b64=_b64(payload),
        signatures=[Signature(kid=kid, alg="ed25519", sig_b64=_b64(sig))]
    )
    return env

def verify_envelope(pub: Ed25519PublicKey, env: Envelope) -> bool:
    payload = _ub64(env.payload_b64)
    if not env.signatures:
        return False
    try:
        sig = _ub64(env.signatures[0].sig_b64)
        pub.verify(sig, payload)
        return True
    except Exception:
        return False