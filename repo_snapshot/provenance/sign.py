# provenance/sign.py
import hashlib, json, os, time
from dataclasses import dataclass
from typing import Optional, Dict, Any
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

STORE_DIR = os.environ.get("IMU_CAS_DIR", "./provenance_store")

os.makedirs(STORE_DIR, exist_ok=True)

@dataclass
class Signature:
    algo: str
    pubkey_pem: str
    ts: float
    sig_hex: str

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def put_blob(content: bytes) -> str:
    digest = _sha256_bytes(content)
    path = os.path.join(STORE_DIR, digest)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(content)
    return digest

def get_blob(digest: str) -> bytes:
    path = os.path.join(STORE_DIR, digest)
    with open(path, "rb") as f:
        return f.read()

def sign_json(priv_pem: bytes, obj: Dict[str, Any]) -> Signature:
    sk = serialization.load_pem_private_key(priv_pem, password=None)
    if not isinstance(sk, Ed25519PrivateKey):
        raise ValueError("only ed25519 private key supported")
    payload = json.dumps(obj, sort_keys=True, separators=(',',':')).encode("utf-8")
    sig = sk.sign(payload)
    pk = sk.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return Signature(
        algo="ed25519",
        pubkey_pem=pk.decode("utf-8"),
        ts=time.time(),
        sig_hex=sig.hex()
    )

def verify_json(sig: Signature, obj: Dict[str, Any]) -> bool:
    payload = json.dumps(obj, sort_keys=True, separators=(',',':')).encode("utf-8")
    pk = serialization.load_pem_public_key(sig.pubkey_pem.encode("utf-8"))
    if not isinstance(pk, Ed25519PublicKey):
        return False
    try:
        pk.verify(bytes.fromhex(sig.sig_hex), payload)
        return True
    except Exception:
        return False

def record_artifact(metadata: Dict[str, Any], content: bytes, signer_priv_pem: bytes) -> Dict[str, Any]:
    digest = put_blob(content)
    record = {
        "digest": digest,
        "size": len(content),
        "kind": metadata.get("kind","generic"),
        "meta": metadata.get("meta",{}),
        "ts": time.time(),
    }
    signature = sign_json(signer_priv_pem, record)
    envelope = {
        "record": record,
        "signature": {
            "algo": signature.algo,
            "pubkey_pem": signature.pubkey_pem,
            "ts": signature.ts,
            "sig_hex": signature.sig_hex
        }
    }
    put_blob(json.dumps(envelope, sort_keys=True).encode("utf-8"))
    return envelope

def require_signed(digest: str) -> Dict[str, Any]:
    # סורק את ה־store ומחפש מעטפה בעלת record.digest = digest
    for fname in os.listdir(STORE_DIR):
        try:
            path = os.path.join(STORE_DIR, fname)
            if not os.path.isfile(path): continue
            blob = open(path, "rb").read()
            if blob.startswith(b"{"):
                env = json.loads(blob.decode("utf-8"))
                rec = env.get("record",{})
                if rec.get("digest")==digest:
                    sig = env.get("signature")
                    ok = verify_json(Signature(**sig), rec)
                    if not ok: raise ValueError("signature invalid for digest="+digest)
                    return env
        except Exception:
            continue
    raise FileNotFoundError("no signed envelope found for digest="+digest)