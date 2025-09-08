# imu_repo/grounded/fact_gate.py
from __future__ import annotations
import time
from typing import List, Dict, Any, Tuple, Optional
from grounded.provenance_store import EvidenceStore
from grounded.validators import Rule, TrustRule, ApiRule, ConsistencyRule, ValidatorRegistry, default_registry, ValidationError
from assurance.errors import ValidationFailed

from grounded.source_policy import policy_singleton as SP
from grounded.provenance import ProvenanceStore

class RefusedNotGrounded(Exception): ...
class GroundingError(Exception): ...


def require_claims(evidence_reqs: List[str], evidence_records: List[Dict[str,Any]], trust_threshold: float | None=None, now: float | None=None):
    """
    Enforce: each required claim has at least one evidence record with trust >= threshold and not expired by source TTL.
    """
    now = time.time() if now is None else now
    thr = SP.trust_threshold if trust_threshold is None else float(trust_threshold)

    # לקבץ לפי claim
    by_claim={}
    for rec in evidence_records:
        c = rec.get("claim")
        by_claim.setdefault(c, []).append(rec)

    missing=[]
    for need in evidence_reqs:
        ok=False
        for r in by_claim.get(need, []):
            trust = float(r.get("trust", 0.0))
            url   = r.get("source_url") or "internal.test://evidence"
            ttl   = SP.ttl_for(url)
            ts    = float(r.get("ts", 0))
            fresh = (ttl==0) or ((now - ts) <= ttl)
            if trust >= thr and fresh and (SP.domain_allowed(url) or url.startswith("internal.test")):
                ok=True
                break
        if not ok:
            missing.append(need)
    if missing:
        raise GroundingError(f"missing_or_weak_evidence:{','.join(missing)}")


class FactGatePolicy:
    def __init__(self, min_sources:int=1, min_trust:float=0.6, max_age_sec:float=86400.0, validators: Optional[Dict[str,str]]=None):
        self.min_sources = min_sources
        self.min_trust   = min_trust
        self.max_age_sec = max_age_sec
        self.validators  = validators or {}  # claim_name -> validator_name


class FactGate:
    """
        Validate claims against evidence index + rules.
         אוכף: אין תשובה בלי טענות + ראיות תקפות.
    """

    def __init__(self, idx:"EvidenceIndex", rules:List["Rule"], store: Optional[EvidenceStore]=None, registry: Optional[ValidatorRegistry]=None,):
        self.idx = idx
        self.rules = rules
        self.store = store or EvidenceStore()
        self.registry = registry or default_registry()


    def check_claims(self, claims:List[Dict[str,Any]], strict:bool=True) -> Tuple[bool,List[Dict[str,Any]]]:
        diagnostics=[]
        all_ok=True
        for claim in claims:
            evid = self.idx.lookup(claim)
            claim_ok=True
            for r in self.rules:
                ok,diag = r.check(claim,evid)
                diagnostics.append(diag)
                if not ok:
                    claim_ok=False
                    if strict:
                        all_ok=False
            if claim_ok and evid:
                diagnostics.append({"rule":"attach_prov","prov_id":evid.get("prov_id"),"ok":True})
        return all_ok, diagnostics
    
    def require_sources(self, sources: List[str]) -> List[Dict[str,Any]]:
        if not sources:
            raise RefusedNotGrounded("grounding_required:no_sources")
        evid = []
        for s in sources:
            rec = self.store.register(s)
            if not self.store.verify(rec["hash"]):
                raise ValidationFailed(f"bad_evidence:{s}")
            evid.append(rec)
        return evid
    
    def check_claim(self, claim: str, now: float, pol: FactGatePolicy) -> List[str]:
        errors: List[str] = []
        evs = self.store.claim_evidences(claim)
        if len(evs) < pol.min_sources:
            errors.append(f"insufficient_sources:{len(evs)}/{pol.min_sources}")
            return errors
        ok_count = 0
        for ev in evs:
            if not self.store.verify_evidence(ev):
                errors.append("bad_signature_or_missing_content")
                continue
            meta = ev.get("meta", {})
            trust = float(meta.get("trust", 0.0))
            if trust < pol.min_trust:
                errors.append(f"low_trust:{trust}")
                continue
            fetched = float(ev.get("fetched_at", 0.0))
            max_age = float(meta.get("max_age_sec", pol.max_age_sec))
            if fetched and max_age>0 and now - fetched > max_age:
                errors.append("stale_evidence")
                continue
            ok_count += 1
        if ok_count < pol.min_sources:
            errors.append(f"not_enough_valid_evidence:{ok_count}/{pol.min_sources}")
        return errors

    def enforce(self, ctx: Dict[str,Any], response_body: Dict[str,Any], policy_map: Optional[Dict[str,Any]]=None) -> None:
        """
        מרים ValidationError אם הבדיקה נכשלת.
        ctx["__claims__"] חייב להכיל [{"claim": str, "sources":[uri,...]}]
        policy_map יכולה להגדיר פוליסות פר-claim; אחרת פוליסה דיפולטית.
        בנוסף: אם יש ולידטור רשום ל-claim — ירוץ על body.
        """
        claims: List[Dict[str,Any]] = ctx.get("__claims__", [])
        if not claims:
            raise ValidationError("no_claims_no_response")

        now=time.time()
        for c in claims:
            name = c.get("claim")
            sources = c.get("sources", [])
            # רשום וקשר ראיות אם טרם קיימות
            evs=[]
            for s in sources:
                try:
                    evs.append(self.store.register_evidence(s))
                except Exception:
                    # אם לא ניתן לרשום מקור — הראיה לא תחשב; הבדיקה תיכשל אם אין מספיק
                    pass
            if evs:
                self.store.link_claim(name, evs)

            # פוליסה
            pol_cfg = (policy_map or {}).get(name, {})
            pol = FactGatePolicy(
                min_sources=int(pol_cfg.get("min_sources", 1)),
                min_trust=float(pol_cfg.get("min_trust", 0.6)),
                max_age_sec=float(pol_cfg.get("max_age_sec", 86400.0)),
                validators=pol_cfg.get("validators")
            )
            errs = self.check_claim(name, now, pol)
            if errs:
                raise ValidationError(f"claim_failed:{name}:{'|'.join(errs)}")

            # ולידטור על גוף התשובה (אם מוגדר עבור claim)
            vmap = pol_cfg.get("validators") or {}
            # ברירת מחדל: validator ששמו = שם הטענה אם קיים
            vname = vmap.get("body") if vmap else name
            try:
                self.registry.run(vname, response_body)
            except ValidationError as e:
                raise ValidationError(f"validator_failed:{name}:{str(e)}")
            

class SchemaRule(Rule):
    
    def check(self, claim:Dict[str,Any], evid:Optional[Dict[str,Any]]):
        ok = isinstance(claim.get("claim"),str) and len(claim["claim"])>0
        return ok, {"rule":"schema","ok":ok}


class UnitRule(Rule):
    
    def check(self, claim:Dict[str,Any], evid:Optional[Dict[str,Any]]):
        return True, {"rule":"unit","ok":True,"skip":True}


class FreshnessRule(Rule):
    
    def __init__(self,max_age_seconds:int=86400): self.max_age=max_age_seconds
    
    def check(self, claim:Dict[str,Any], evid:Optional[Dict[str,Any]]):
        if not evid:
            return False, {"rule":"freshness","ok":False,"reason":"no_evidence"}
        age=time.time()-evid.get("ts",0)
        ok= age <= self.max_age
        return ok, {"rule":"freshness","ok":ok,"age":age,"max":self.max_age}

class EvidenceIndex:
    """Index claims to evidence stored in provenance store."""
    
    def __init__(self, store:ProvenanceStore):
        self.store=store
    
    def lookup(self, claim:Dict[str,Any]) -> Optional[Dict[str,Any]]:
        return self.store.get_by_claim(claim.get("claim"))
