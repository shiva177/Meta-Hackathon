"""
Deterministic keyword-based policy lookup engine.
No ML, no embeddings — pure token intersection for reproducible scoring.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from .models import PolicyResult

# ---------------------------------------------------------------------------
# Data model for internal use
# ---------------------------------------------------------------------------

class PolicyDocument:
    def __init__(self, policy_id: str, title: str, keywords: List[str], body: str) -> None:
        self.policy_id = policy_id
        self.title = title
        self.keywords = [k.lower() for k in keywords]
        self.body = body


# ---------------------------------------------------------------------------
# Load policies once at import time
# ---------------------------------------------------------------------------

_POLICY_DB: List[PolicyDocument] = []

def _load_policies() -> List[PolicyDocument]:
    path = Path(__file__).parent / "data" / "policy_kb.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        PolicyDocument(
            policy_id=doc["policy_id"],
            title=doc["title"],
            keywords=doc["keywords"],
            body=doc["body"],
        )
        for doc in raw
    ]


def _get_db() -> List[PolicyDocument]:
    global _POLICY_DB
    if not _POLICY_DB:
        _POLICY_DB = _load_policies()
    return _POLICY_DB


# ---------------------------------------------------------------------------
# Lookup function
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set:
    """Lowercase, split on non-word chars, filter short tokens."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return {t for t in tokens if len(t) > 2}


def lookup_policy(query: str) -> PolicyResult:
    """
    Deterministic keyword-based policy lookup.
    Returns the policy with the highest keyword overlap with the query.
    Ties broken by document order (first match wins).
    confidence = overlap_count / len(policy.keywords)
    """
    db = _get_db()
    query_tokens = _tokenize(query)

    best_doc: Optional[PolicyDocument] = None
    best_score: float = 0.0
    best_overlap: int = 0

    for doc in db:
        kw_set = set(doc.keywords)
        overlap = len(query_tokens & kw_set)
        if len(kw_set) == 0:
            continue
        score = overlap / len(kw_set)
        # Strict improvement required to avoid ties changing order
        if overlap > best_overlap or (overlap == best_overlap and score > best_score):
            best_overlap = overlap
            best_score = score
            best_doc = doc

    if best_doc is None or best_overlap == 0:
        return PolicyResult(
            query=query,
            matched_policy_id=None,
            policy_title=None,
            policy_body=None,
            confidence=0.0,
        )

    return PolicyResult(
        query=query,
        matched_policy_id=best_doc.policy_id,
        policy_title=best_doc.title,
        policy_body=best_doc.body,
        confidence=round(best_score, 4),
    )
