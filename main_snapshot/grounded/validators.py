# imu_repo/grounded/validators.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import json, time
from grounded.provenance_store import CAS
from adapters.http_fetch import http_fetch


class ValidationError(Exception): ...


def type_of(x: Any) -> str:
    if isinstance(x, bool): return "bool"
    if isinstance(x, int): return "int"
    if isinstance(x, float): return "float"
    if isinstance(x, str): return "string"
    if isinstance(x, list): return "array"
    if isinstance(x, dict): return "object"
    if x is None: return "null"
    return type(x).__name__


def validate_schema(data: Any, schema: Dict[str,Any]) -> None:
    t = schema.get("type")
    if t=="object":
        if not isinstance(data, dict): raise ValidationError("type:object")
        req = schema.get("required", [])
        for r in req:
            if r not in data: raise ValidationError(f"required:{r}")
        props = schema.get("properties", {})
        for k in props:
            if k in data:
                validate_schema(data[k], props[k])
    elif t=="array":
        if not isinstance(data, list): raise ValidationError("type:array")
        items = schema.get("items")
        if items:
            for it in data: validate_schema(it, items)
    elif t in ("number","float"):
        if not isinstance(data,(int,float)): raise ValidationError("type:number")
        if "min" in schema and data < schema["min"]: raise ValidationError("min")
        if "max" in schema and data > schema["max"]: raise ValidationError("max")
    elif t=="integer":
        if not isinstance(data,int): raise ValidationError("type:integer")
        if "min" in schema and data < schema["min"]: raise ValidationError("min")
        if "max" in schema and data > schema["max"]: raise ValidationError("max")
    elif t=="string":
        if not isinstance(data,str): raise ValidationError("type:string")
        if "enum" in schema and data not in schema["enum"]: raise ValidationError("enum")
    elif t is None:
        return
    else:
        raise ValidationError(f"unknown_type:{t}")


def validate_unit(value: float, unit: str) -> None:
    unit = unit.lower()
    if unit in ("ms","millisecond","milliseconds"):
        if value < 0: raise ValidationError("unit:ms<0")
    elif unit in ("s","sec","second","seconds"):
        if value < 0: raise ValidationError("unit:s<0")
    elif unit in ("kb","kilobyte"):
        if value < 0: raise ValidationError("unit:kb<0")
    elif unit in ("mb","megabyte"):
        if value < 0: raise ValidationError("unit:mb<0")
    else:
        raise ValidationError(f"unknown_unit:{unit}")


def validate_dependencies(data: Dict[str,Any], deps: Dict[str, List[str]]) -> None:
    for k,need in deps.items():
        if k in data:
            for d in need:
                if d not in data:
                    raise ValidationError(f"dep:{k}->{d}")


class ValidatorRegistry:
    def __init__(self):
        self.registry: Dict[str, Dict[str,Any]] = {}

    def register(self, name: str, schema: Dict[str,Any], units: Optional[Dict[str,str]]=None, deps: Optional[Dict[str,List[str]]]=None):
        self.registry[name] = {"schema": schema, "units": units or {}, "deps": deps or {}}

    def run(self, name: str, data: Dict[str,Any]) -> None:
        if name not in self.registry: raise ValidationError(f"no_validator:{name}")
        spec = self.registry[name]
        validate_schema(data, spec["schema"])
        if spec["deps"]:
            validate_dependencies(data, spec["deps"])
        for k,u in spec["units"].items():
            if k in data: validate_unit(float(data[k]), u)

# ברירת־מחדל:
_default = ValidatorRegistry()
_default.register(
    "sum_result",
    schema={"type":"object","required":["sum"],"properties":{"sum":{"type":"number","min":-1e12,"max":1e12}}},
)
_default.register(
    "fs_echo",
    schema={"type":"object","required":["echo"],"properties":{"echo":{"type":"string"}}},
)
_default.register(
    "http_doc",
    schema={"type":"object","required":["title","version"],"properties":{
        "title":{"type":"string"},
        "version":{"type":"string"},
        "age_sec":{"type":"number","min":0}
    }},
    units={"age_sec":"s"},
    deps={"age_sec":["version"]}
)


def default_registry() -> ValidatorRegistry:
    return _default


class Rule:
    def check(self, claim:Dict[str,Any], evid:Optional[Dict[str,Any]]) -> Tuple[bool,Dict[str,Any]]:
        raise NotImplementedError


class TrustRule(Rule):
    """Require minimal trust score on evidence (0..1)."""
    def __init__(self, min_trust: float = 0.6): self.min_trust=min_trust
    def check(self, claim, evid):
        if not evid: return False, {"rule":"trust","ok":False,"reason":"no_evidence"}
        ok = float(evid.get("trust",0.0)) >= self.min_trust
        return ok, {"rule":"trust","ok":ok,"trust":float(evid.get("trust",0.0)),"min":self.min_trust}


class ApiRule(Rule):
    """
    If evidence includes a source like 'api:https://host/path?query...',
    fetch and validate a simple JSON predicate (payload.expected == true or field equals).
    This is generic-but-real: requires actual network & allowlist at the engine call-site.
    """
    def __init__(self, allow_hosts: Optional[List[str]] = None, timeout: float = 5.0):
        self.allow_hosts = allow_hosts or []
        self.timeout = timeout

    def check(self, claim, evid):
        if not evid: return False, {"rule":"api","ok":False,"reason":"no_evidence"}
        sources: list[str] = evid.get("sources", [])
        apis = [s[4:] for s in sources if isinstance(s,str) and s.startswith("api:")]
        if not apis:
            return True, {"rule":"api","ok":True,"skip":True}
        # naive validation: expect {"ok": true} or {"claim": "...", "valid": true}
        for url in apis:
            try:
                resp = http_fetch(url, timeout=self.timeout, allow_hosts=self.allow_hosts)
                if resp["status"]//100 != 2: 
                    return False, {"rule":"api","ok":False,"status":resp["status"],"url":url}
                try:
                    data = json.loads(resp["body"])
                except Exception:
                    # allow text "OK"
                    data = {"ok": resp["body"].decode("utf-8",errors="ignore").strip().upper()=="OK"}
                if (isinstance(data,dict) and (data.get("ok") is True or data.get("valid") is True
                    or data.get("claim")==claim.get("claim") and data.get("valid") is True)):
                    return True, {"rule":"api","ok":True,"url":url}
                return False, {"rule":"api","ok":False,"url":url,"reason":"predicate_failed"}
            except Exception as e:
                return False, {"rule":"api","ok":False,"url":url,"error":str(e)}
        return False, {"rule":"api","ok":False,"reason":"no_api_validated"}


class ConsistencyRule(Rule):
    """If evidence payload declares 'verified': True, accept; else fail."""
    def check(self, claim, evid):
        if not evid: return False, {"rule":"consistency","ok":False,"reason":"no_evidence"}
        ok = bool(evid.get("payload",{}).get("verified", False))
        return ok, {"rule":"consistency","ok":ok}
