# governance/policy.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List

@dataclass
class EvidenceRule:
    min_trust: float = 0.9
    max_age_sec: int = 24*3600
    allowed_domains: List[str] = field(default_factory=lambda: ["example.com"])
    require_signature: bool = True

@dataclass
class RespondPolicy:
    # האם חייבים claims+evidence לכל תשובה
    require_claims: bool = True
    require_evidence: bool = True
    # האם מותר חשבון מתמטי מדויק ללא claims (למשל 2+2)
    allow_math_without_claims: bool = False
    # סף ומגבלות
    max_claims: int = 64
    # כללי עדויות
    evidence: EvidenceRule = field(default_factory=EvidenceRule)

@dataclass
class Subspace:
    """שכבת תצורה פר־משתמש לפי ארגון/תפקיד/פרויקט."""
    role: str = "user"
    org: str = "public"
    project: str = "default"