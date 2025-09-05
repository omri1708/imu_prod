# provenance/store.py
# -*- coding: utf-8 -*-
import os, time, json, hashlib, base64
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from adapters.contracts import ResourceRequired
import hashlib, json, os, shutil, time
from pathlib import Path
from typing import Optional, Dict
from contracts.base import Artifact

class ProvenanceStore:
    """
    Content-addressable store:
      - כל artifact מקבל sha256 לפי תוכנו.
      - נשמר metadata.json עם רמות אמון/תיעוד מקור, חותמות זמן, וחוזים רלוונטיים.
    """
    def __init__(self, root: str = ".imu_provenance"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _sha256_file(self, p: Path) -> str:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    def add(self, art: Artifact, trust_level: str = "unverified", evidence: Optional[Dict]=None) -> Artifact:
        p = Path(art.path)
        if not p.exists():
            raise FileNotFoundError(f"artifact_not_found: {p}")
        digest = self._sha256_file(p)
        dst_dir = self.root / digest
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / p.name
        shutil.copy2(p, dst_path)
        meta = {
            "kind": art.kind,
            "filename": p.name,
            "sha256": digest,
            "time": time.time(),
            "trust_level": trust_level,
            "evidence": evidence or {},
            "metadata": art.metadata or {},
        }
        (dst_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        art.provenance_sha256 = digest
        return art

    def get(self, sha256: str) -> Path:
        d = self.root / sha256
        if not d.exists():
            raise FileNotFoundError(f"missing_digest: {sha256}")
        # Return path to the stored payload (first non-metadata file)
        for child in d.iterdir():
            if child.name != "metadata.json":
                return child
        raise FileNotFoundError(f"no_payload_for_digest: {sha256}")

def _ensure_keys(key_dir: str):
    try:
        from nacl.signing import SigningKey
    except Exception:
        raise ResourceRequired("PyNaCl (ed25519)", "pip install pynacl")
    os.makedirs(key_dir, exist_ok=True)
    skf = os.path.join(key_dir, "ed25519.sk")
    pkf = os.path.join(key_dir, "ed25519.pk")
    if not os.path.exists(skf):
        sk = SigningKey.generate()
        with open(skf, "wb") as f: f.write(sk.encode())
        with open(pkf, "wb") as f: f.write(sk.verify_key.encode())
    else:
        with open(skf, "rb") as f: sk = SigningKey(f.read())
    return sk

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@dataclass
class EvidenceMeta:
    url: Optional[str]
    fetched_ts: float
    sha256: str
    trust: float  # 0..1
    not_before_ts: Optional[float] = None
    not_after_ts: Optional[float] = None
    sig_b64: Optional[str] = None

class CASStore:
    def __init__(self, root_dir: str = ".imu/cas", key_dir: str = ".imu/keys"):
        self.root_dir = root_dir
        self.key_dir = key_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _path(self, digest: str) -> str:
        return os.path.join(self.root_dir, digest)

    def put_bytes(self, b: bytes, sign: bool = True, url: str = None, trust: float = 0.5,
                  not_after_days: int = 365) -> EvidenceMeta:
        h = _hash_bytes(b)
        p = self._path(h)
        if not os.path.exists(p):
            with open(p, "wb") as f: f.write(b)
        meta = EvidenceMeta(url=url, fetched_ts=time.time(), sha256=h, trust=float(trust))
        if not_after_days:
            meta.not_after_ts = meta.fetched_ts + not_after_days*24*3600
        if sign:
            try:
                from nacl.signing import SigningKey
            except Exception:
                raise ResourceRequired("PyNaCl (ed25519)", "pip install pynacl")
            sk = _ensure_keys(self.key_dir)
            sig = sk.sign(h.encode("utf-8")).signature
            meta.sig_b64 = base64.b64encode(sig).decode("ascii")
        # שמירת מטא-דאטה
        with open(p + ".json", "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, ensure_ascii=False, indent=2)
        return meta

    def get(self, digest: str) -> Optional[EvidenceMeta]:
        p = self._path(digest) + ".json"
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return EvidenceMeta(**d)

    def verify_meta(self, meta: EvidenceMeta) -> bool:
        # תוקף זמן + חתימה
        now = time.time()
        if meta.not_before_ts and now < meta.not_before_ts:
            return False
        if meta.not_after_ts and now > meta.not_after_ts:
            return False
        if meta.sig_b64:
            try:
                from nacl.signing import VerifyKey
            except Exception:
                raise ResourceRequired("PyNaCl (ed25519)", "pip install pynacl")
            pkf = os.path.join(self.key_dir, "ed25519.pk")
            if not os.path.exists(pkf):
                return False
            with open(pkf, "rb") as f: vk = VerifyKey(f.read())
            try:
                vk.verify(meta.sha256.encode("utf-8"), base64.b64decode(meta.sig_b64))
            except Exception:
                return False
        return True