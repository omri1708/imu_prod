# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Optional
import base64, hmac, hashlib, os

class Signer:
    """
    Pluggable signer. Prefers Ed25519 via cryptography; falls back to HMAC-SHA256.
    """
    def __init__(self, key_id: str = "default", ed25519_priv_pem: Optional[bytes] = None,
                 hmac_key: Optional[bytes] = None):
        self.key_id = key_id
        self._mode = None
        self._ed25519_private = None
        if ed25519_priv_pem:
            try:
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.primitives.asymmetric import ed25519
                self._ed25519_private = serialization.load_pem_private_key(ed25519_priv_pem, password=None)
                self._mode = "ed25519"
            except Exception as e:
                raise RuntimeError(f"Failed to load ed25519 key: {e}")
        elif hmac_key:
            self._mode = "hmac"
            self._hmac_key = hmac_key
        else:
            # fallback dev-key HMAC (not for production)
            self._mode = "hmac"
            self._hmac_key = os.environ.get("ASSURANCE_HMAC_KEY", "dev-key").encode("utf-8")

    def sign(self, data: bytes) -> Dict[str, Any]:
        if self._mode == "ed25519":
            from cryptography.hazmat.primitives.asymmetric import ed25519
            sig = self._ed25519_private.sign(data)
            return {"alg": "Ed25519", "kid": self.key_id, "sig_b64": base64.b64encode(sig).decode("ascii")}
        else:
            mac = hmac.new(self._hmac_key, data, hashlib.sha256).digest()
            return {"alg": "HMAC-SHA256", "kid": self.key_id, "sig_b64": base64.b64encode(mac).decode("ascii")}
