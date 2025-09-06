# provenance/keyring.py
# Ed25519 keyring with rotation, list/export/import, and on-disk storage.
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from pathlib import Path
import json, time, secrets

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey
)
from cryptography.hazmat.primitives import serialization

@dataclass
class KeyMeta:
    kid: str           # key id
    created: float
    active: bool
    comment: str = ""

class Keyring:
    def __init__(self, root: str = ".imu/keys"):
        self.root = Path(root)
        (self.root / "priv").mkdir(parents=True, exist_ok=True)
        (self.root / "pub").mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.index: Dict[str, KeyMeta] = {}
        self._load_index()

    def _load_index(self):
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.index = {kid: KeyMeta(**meta) for kid, meta in data.items()}
        else:
            self._persist()

    def _persist(self):
        self.index_path.write_text(
            json.dumps({k: asdict(v) for k, v in self.index.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def list(self) -> List[KeyMeta]:
        return list(self.index.values())

    def current_kid(self) -> Optional[str]:
        for kid, meta in self.index.items():
            if meta.active:
                return kid
        return None

    def _gen_kid(self) -> str:
        return secrets.token_hex(8)

    def generate(self, comment: str = "") -> KeyMeta:
        sk = Ed25519PrivateKey.generate()
        pk = sk.public_key()
        kid = self._gen_kid()
        # write PEMs
        priv_pem = sk.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub_pem = pk.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        (self.root / "priv" / f"{kid}.pem").write_bytes(priv_pem)
        (self.root / "pub" / f"{kid}.pem").write_bytes(pub_pem)
        # deactivate existing
        for meta in self.index.values():
            meta.active = False
        meta = KeyMeta(kid=kid, created=time.time(), active=True, comment=comment)
        self.index[kid] = meta
        self._persist()
        return meta

    def rotate(self, comment: str = "rotation") -> KeyMeta:
        return self.generate(comment=comment)

    def set_active(self, kid: str):
        if kid not in self.index:
            raise ValueError("unknown kid")
        for m in self.index.values():
            m.active = False
        self.index[kid].active = True
        self._persist()

    def export_public_keys(self) -> Dict[str, str]:
        out = {}
        for kid in self.index:
            out[kid] = (self.root / "pub" / f"{kid}.pem").read_text(encoding="utf-8")
        return out

    def load_private(self, kid: Optional[str] = None) -> Ed25519PrivateKey:
        if kid is None:
            kid = self.current_kid()
            if not kid:
                raise RuntimeError("no active key")
        pem = (self.root / "priv" / f"{kid}.pem").read_bytes()
        return serialization.load_pem_private_key(pem, password=None)  # type: ignore

    def load_public(self, kid: str) -> Ed25519PublicKey:
        pem = (self.root / "pub" / f"{kid}.pem").read_bytes()
        return serialization.load_pem_public_key(pem)  # type: ignore