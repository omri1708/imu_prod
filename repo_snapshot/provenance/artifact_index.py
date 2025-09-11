# provenance/artifact_index.py
# Mapping between Docker image -> digest -> provenance envelope path/metadata.
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import json, time, re

@dataclass
class ImageRecord:
    image: str
    digest: str
    envelope_path: Optional[str] = None
    ts: float = time.time()
    meta: Dict[str, Any] = None

class ArtifactIndex:
    def __init__(self, path: str = ".imu/artifacts/index.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._idx: Dict[str, ImageRecord] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._idx = {k: ImageRecord(**v) for k,v in data.items()}
        else:
            self._persist()

    def _persist(self):
        self.path.write_text(json.dumps({k: asdict(v) for k,v in self._idx.items()}, ensure_ascii=False, indent=2), encoding="utf-8")

    def put(self, image: str, digest: str, envelope_path: Optional[str] = None, meta: Dict[str,Any]=None):
        key = f"{image}@{digest}"
        self._idx[key] = ImageRecord(image=image, digest=digest, envelope_path=envelope_path, meta=meta or {})
        self._persist()

    def find(self, image: str) -> List[ImageRecord]:
        pat = re.escape(image)+"@"
        return [v for k,v in self._idx.items() if k.startswith(pat)]

    def by_digest(self, digest: str) -> Optional[ImageRecord]:
        for v in self._idx.values():
            if v.digest == digest:
                return v
        return None

INDEX = ArtifactIndex()