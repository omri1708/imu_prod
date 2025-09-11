# imu_repo/persistence/policy_store.py
from __future__ import annotations
from typing import Dict, Any, Optional
import os, json, time, hashlib, shutil

class PolicyStore:
    """
    שומר גרסאות 'מדיניות' (Policy) עם מטא-דאטה:
    - policy.json (הפעילה)
    - versions/{version}.json
    """
    def __init__(self, root:str=".imu_state/policy"):
        self.root=root
        os.makedirs(self.root, exist_ok=True)
        self.active=os.path.join(self.root,"policy.json")
        self.versions=os.path.join(self.root,"versions")
        os.makedirs(self.versions, exist_ok=True)
        if not os.path.exists(self.active):
            self._write(self.active, {"version": "v0", "meta":{"created":time.time()}, "config":{}})

    def _write(self, path:str, obj:Dict[str,Any]):
        with open(path,"w",encoding="utf-8") as f:
            json.dump(obj,f,ensure_ascii=False,indent=2)

    def _read(self, path:str) -> Dict[str,Any]:
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)

    def current(self)->Dict[str,Any]:
        return self._read(self.active)

    def stage(self, candidate_cfg: Dict[str,Any], note:str="")->str:
        ver=f"v{int(time.time())}"
        obj={"version":ver,"meta":{"created":time.time(),"note":note},"config":candidate_cfg}
        self._write(os.path.join(self.versions,f"{ver}.json"), obj)
        return ver

    def promote(self, version:str)->Dict[str,Any]:
        path=os.path.join(self.versions,f"{version}.json")
        obj=self._read(path)
        self._write(self.active, obj)
        return obj

    def rollback(self, to_version:str)->Dict[str,Any]:
        return self.promote(to_version)
