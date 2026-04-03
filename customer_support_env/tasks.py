"""
Task definitions and scenario generation for the Customer Support environment.
No I/O beyond reading tickets.json at load time.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import Ticket, TriageItem

# ---------------------------------------------------------------------------
# Priority matrix — single source of truth used by both tasks.py and graders.py
# ---------------------------------------------------------------------------

PRIORITY_MATRIX: Dict[Tuple[str, str], str] = {
    ("enterprise", "billing"):         "critical",
    ("enterprise", "technical"):       "high",
    ("enterprise", "account"):         "high",
    ("pro",        "billing"):         "high",
    ("pro",        "technical"):       "medium",
    ("pro",        "account"):         "medium",
    ("free",       "billing"):         "medium",
    ("free",       "technical"):       "low",
    ("free",       "account"):         "low",
    ("enterprise", "feature_request"): "low",
    ("pro",        "feature_request"): "low",
    ("free",       "feature_request"): "low",
    ("enterprise", "general"):         "medium",
    ("pro",        "general"):         "medium",
    ("free",       "general"):         "low",
}

# Abuse is always critical regardless of tier
_ABUSE_PRIORITY = "critical"


def get_expected_priority(tier: str, category: str) -> str:
    """Return expected priority for a given tier + category combo."""
    if category == "abuse":
        return _ABUSE_PRIORITY
    return PRIORITY_MATRIX.get((tier, category), "low")


# ---------------------------------------------------------------------------
# Ticket corpus loader
# ---------------------------------------------------------------------------

_TICKET_CORPUS: List[dict] = []


def _load_corpus() -> List[dict]:
    global _TICKET_CORPUS
    if not _TICKET_CORPUS:
        path = Path(__file__).parent / "data" / "tickets.json"
        with open(path, "r", encoding="utf-8") as f:
            _TICKET_CORPUS = json.load(f)
    return _TICKET_CORPUS


def _make_ticket(raw: dict) -> Ticket:
    """Strip internal ground-truth fields before exposing to agent."""
    return Ticket(
        ticket_id=raw["ticket_id"],
        subject=raw["subject"],
        body=raw["body"],
        customer_tier=raw["customer_tier"],
        category_hint=None,            # always hidden from agent
        created_at=raw["created_at"],
        channel=raw["channel"],
        previous_contacts=raw.get("previous_contacts", 0),
        account_age_days=raw.get("account_age_days", 0),
    )


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    # Easy
    correct_category: Optional[str] = None
    correct_priority: Optional[str] = None
    required_response_keywords: Optional[List[str]] = None

    # Medium
    correct_triage: Optional[List[TriageItem]] = None

    # Hard
    escalation_required: bool = False
    correct_escalation_tier: Optional[str] = None
    required_policy_id: Optional[str] = None
    correct_resolution_note_keywords: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    task_id: str
    description: str
    max_steps: int
    instructions: str
    n_tickets: int


TASK_SPECS: Dict[str, TaskSpec] = {
    "easy": TaskSpec(
        task_id="easy",
        description="Classify and respond to a single customer support ticket.",
        max_steps=6,
        n_tickets=1,
        instructions=(
            "TASK: You are a customer support agent. You will receive ONE support ticket.\n"
            "Your goal is to:\n"
            "  1. classify() — identify the correct category for the ticket\n"
            "  2. set_priority() — assign the correct priority level\n"
            "  3. respond() — write a professional response to the customer\n"
            "  4. resolve() — mark the ticket as resolved with a resolution note\n\n"
            "Available action types and required fields:\n"
            "  classify: {action_type: 'classify', category: '<billing|technical|account|feature_request|abuse|general>'}\n"
            "  set_priority: {action_type: 'set_priority', priority: '<critical|high|medium|low>'}\n"
            "  respond: {action_type: 'respond', response_text: '<your message to the customer>'}\n"
            "  resolve: {action_type: 'resolve', resolved_ticket_id: '<ticket_id>', resolution_note: '<internal note>'}\n\n"
            "Complete all four steps to maximize your score. Efficiency bonus for finishing in ≤ 3 steps."
        ),
    ),
    "medium": TaskSpec(
        task_id="medium",
        description="Triage a queue of 5 tickets by assigning correct categories and priorities.",
        max_steps=8,
        n_tickets=5,
        instructions=(
            "TASK: You are a support manager. You have 5 tickets in the queue.\n"
            "Your goal is to triage ALL tickets at once using the bulk_triage action.\n"
            "For each ticket, assign the correct category and priority based on:\n"
            "  - Customer tier (enterprise > pro > free)\n"
            "  - Issue type (abuse and enterprise billing = critical, etc.)\n\n"
            "Priority rules:\n"
            "  - abuse: always CRITICAL regardless of tier\n"
            "  - enterprise + billing: critical | enterprise + technical: high | enterprise + account: high\n"
            "  - pro + billing: high | pro + technical: medium | pro + account: medium\n"
            "  - free + billing: medium | free + technical: low | free + account: low\n"
            "  - feature_request: always LOW | general: medium (pro/enterprise), low (free)\n\n"
            "Action to use:\n"
            "  bulk_triage: {\n"
            "    action_type: 'bulk_triage',\n"
            "    triage_list: [\n"
            "      {ticket_id: '<id>', category: '<category>', priority: '<priority>'},\n"
            "      ... (one entry per ticket)\n"
            "    ]\n"
            "  }\n\n"
            "You may use lookup_policy to check the escalation policy before triaging.\n"
            "You can submit bulk_triage multiple times — only the last submission is graded.\n"
            "Efficiency bonus for finishing in ≤ 3 steps."
        ),
    ),
    "hard": TaskSpec(
        task_id="hard",
        description=(
            "Resolve a complex enterprise ticket requiring policy lookup, "
            "correct escalation, and a thorough resolution."
        ),
        max_steps=12,
        n_tickets=1,
        instructions=(
            "TASK: You are a senior support specialist. You have a COMPLEX enterprise ticket.\n"
            "This ticket requires a multi-step resolution process:\n"
            "  1. classify() — identify the ticket category\n"
            "  2. set_priority() — assign priority (enterprise billing disputes are critical)\n"
            "  3. lookup_policy() — search the policy knowledge base for relevant procedures\n"
            "  4. escalate() — escalate to the appropriate support tier with a clear reason\n"
            "  5. respond() — send a professional response to the customer\n"
            "  6. resolve() — close the ticket with a thorough resolution note\n\n"
            "Available action types:\n"
            "  classify: {action_type: 'classify', category: '...'}\n"
            "  set_priority: {action_type: 'set_priority', priority: '...'}\n"
            "  lookup_policy: {action_type: 'lookup_policy', policy_query: '<search terms>'}\n"
            "  escalate: {action_type: 'escalate', escalation_tier: '<tier2|tier3|legal|executive>', escalation_reason: '<reason>'}\n"
            "  respond: {action_type: 'respond', response_text: '<message>'}\n"
            "  resolve: {action_type: 'resolve', resolved_ticket_id: '<ticket_id>', resolution_note: '<note>'}\n\n"
            "Scoring is based on completing all required steps correctly.\n"
            "Enterprise billing disputes over $1000 require tier3 escalation per policy.\n"
            "Use lookup_policy before escalating to find the correct procedure."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Task instance
# ---------------------------------------------------------------------------

@dataclass
class TaskInstance:
    spec: TaskSpec
    tickets: List[Ticket]
    ground_truth: GroundTruth
    seed: int
    raw_tickets: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Task loaders
# ---------------------------------------------------------------------------

def _load_easy(rng: random.Random) -> TaskInstance:
    corpus = _load_corpus()
    # Pick a ticket with a clear category (non-abuse, non-enterprise for easy difficulty)
    candidates = [
        t for t in corpus
        if t.get("_correct_category") in ("billing", "technical", "account", "general")
        and t.get("customer_tier") in ("free", "pro")
    ]
    raw = rng.choice(candidates)
    ticket = _make_ticket(raw)
    category = raw["_correct_category"]
    tier = raw["customer_tier"]
    priority = get_expected_priority(tier, category)

    gt = GroundTruth(
        correct_category=category,
        correct_priority=priority,
        required_response_keywords=_response_keywords_for(category),
    )
    return TaskInstance(
        spec=TASK_SPECS["easy"],
        tickets=[ticket],
        ground_truth=gt,
        seed=rng.randint(0, 10**9),
        raw_tickets=[raw],
    )


def _load_medium(rng: random.Random) -> TaskInstance:
    corpus = _load_corpus()
    # Pick 5 diverse tickets covering multiple tiers and categories
    # Ensure at least 1 abuse, 1 enterprise, 1 billing
    abuse_pool = [t for t in corpus if t.get("_correct_category") == "abuse"]
    enterprise_pool = [t for t in corpus if t.get("customer_tier") == "enterprise" and t.get("_correct_category") != "abuse"]
    billing_pool = [t for t in corpus if t.get("_correct_category") == "billing" and t.get("customer_tier") != "enterprise"]
    general_pool = [t for t in corpus if t.get("_correct_category") in ("general", "feature_request", "technical", "account")]

    chosen: List[dict] = []
    seen_ids: set = set()

    def pick_unique(pool: List[dict]) -> Optional[dict]:
        shuffled = pool[:]
        rng.shuffle(shuffled)
        for item in shuffled:
            if item["ticket_id"] not in seen_ids:
                seen_ids.add(item["ticket_id"])
                return item
        return None

    # Mandatory: 1 abuse
    item = pick_unique(abuse_pool)
    if item:
        chosen.append(item)

    # Mandatory: 1 enterprise non-abuse
    item = pick_unique(enterprise_pool)
    if item:
        chosen.append(item)

    # Mandatory: 1 billing (non-enterprise)
    item = pick_unique(billing_pool)
    if item:
        chosen.append(item)

    # Fill remaining slots from general pool
    while len(chosen) < 5:
        item = pick_unique(general_pool)
        if item is None:
            # fallback: any remaining ticket
            remaining = [t for t in corpus if t["ticket_id"] not in seen_ids]
            if not remaining:
                break
            rng.shuffle(remaining)
            item = remaining[0]
            seen_ids.add(item["ticket_id"])
        chosen.append(item)

    # Shuffle final list so order is non-deterministic
    rng.shuffle(chosen)

    tickets = [_make_ticket(raw) for raw in chosen]
    correct_triage = [
        TriageItem(
            ticket_id=raw["ticket_id"],
            category=raw["_correct_category"],
            priority=get_expected_priority(raw["customer_tier"], raw["_correct_category"]),
        )
        for raw in chosen
    ]

    gt = GroundTruth(correct_triage=correct_triage)
    return TaskInstance(
        spec=TASK_SPECS["medium"],
        tickets=tickets,
        ground_truth=gt,
        seed=rng.randint(0, 10**9),
        raw_tickets=chosen,
    )


def _load_hard(rng: random.Random) -> TaskInstance:
    corpus = _load_corpus()
    # Hard task: enterprise billing dispute with multiple prior contacts
    candidates = [
        t for t in corpus
        if t.get("customer_tier") == "enterprise"
        and t.get("_correct_category") == "billing"
        and t.get("previous_contacts", 0) >= 2
    ]
    if not candidates:
        # Fallback: any enterprise billing
        candidates = [
            t for t in corpus
            if t.get("customer_tier") == "enterprise"
            and t.get("_correct_category") == "billing"
        ]
    raw = rng.choice(candidates)
    ticket = _make_ticket(raw)

    gt = GroundTruth(
        correct_category="billing",
        correct_priority="critical",
        escalation_required=True,
        correct_escalation_tier="tier3",
        required_policy_id="POL-BILLING-002",   # Enterprise Contract and Invoice Disputes
        correct_resolution_note_keywords=["refund", "escalated", "account manager"],
    )
    return TaskInstance(
        spec=TASK_SPECS["hard"],
        tickets=[ticket],
        ground_truth=gt,
        seed=rng.randint(0, 10**9),
        raw_tickets=[raw],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_task(task_id: str, seed: Optional[int] = None) -> TaskInstance:
    """
    Load a task instance.
    Uses a local Random instance to avoid polluting global random state.
    """
    actual_seed = seed if seed is not None else random.randint(0, 10**9)
    rng = random.Random(actual_seed)

    if task_id == "easy":
        instance = _load_easy(rng)
    elif task_id == "medium":
        instance = _load_medium(rng)
    elif task_id == "hard":
        instance = _load_hard(rng)
    else:
        raise ValueError(f"Unknown task_id: {task_id!r}. Must be 'easy', 'medium', or 'hard'.")

    instance.seed = actual_seed
    return instance


def _response_keywords_for(category: str) -> List[str]:
    """Return required keywords for a response to a given category."""
    keywords_map = {
        "billing":         ["refund", "charge"],
        "technical":       ["issue", "team"],
        "account":         ["account", "reset"],
        "feature_request": ["feedback", "team"],
        "abuse":           ["action", "review"],
        "general":         ["help", "let us know"],
    }
    return keywords_map.get(category, ["thank", "help"])
