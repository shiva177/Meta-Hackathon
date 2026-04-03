"""
Pydantic models for Customer Support Ticket Resolution Environment.
All models are the canonical source of truth for field names and types.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ticket
# ---------------------------------------------------------------------------

class Ticket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: Literal["free", "pro", "enterprise"]
    category_hint: Optional[str] = None   # None when sent to agent; used internally for grading
    created_at: str                        # ISO-8601 string
    channel: Literal["email", "chat", "phone"]
    previous_contacts: int = 0
    account_age_days: int = 0


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TriageItem(BaseModel):
    ticket_id: str
    category: Literal["billing", "technical", "account", "feature_request", "abuse", "general"]
    priority: Literal["critical", "high", "medium", "low"]


class Action(BaseModel):
    action_type: Literal[
        "classify",
        "set_priority",
        "respond",
        "escalate",
        "lookup_policy",
        "resolve",
        "bulk_triage",
    ]

    # --- classify ---
    category: Optional[
        Literal["billing", "technical", "account", "feature_request", "abuse", "general"]
    ] = None

    # --- set_priority ---
    priority: Optional[Literal["critical", "high", "medium", "low"]] = None

    # --- respond ---
    response_text: Optional[str] = None
    response_template_id: Optional[str] = None

    # --- escalate ---
    escalation_tier: Optional[Literal["tier2", "tier3", "legal", "executive"]] = None
    escalation_reason: Optional[str] = None

    # --- lookup_policy ---
    policy_query: Optional[str] = None

    # --- resolve ---
    resolution_note: Optional[str] = None
    resolved_ticket_id: Optional[str] = None

    # --- bulk_triage (medium task only) ---
    triage_list: Optional[List[TriageItem]] = None


# ---------------------------------------------------------------------------
# Policy lookup result
# ---------------------------------------------------------------------------

class PolicyResult(BaseModel):
    query: str
    matched_policy_id: Optional[str] = None
    policy_title: Optional[str] = None
    policy_body: Optional[str] = None
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    step: int
    task_id: str
    tickets: List[Ticket]
    active_ticket_id: Optional[str] = None
    policy_result: Optional[PolicyResult] = None
    action_feedback: Optional[str] = None
    action_valid: Optional[bool] = None
    escalation_queue: List[str] = Field(default_factory=list)
    resolved_tickets: List[str] = Field(default_factory=list)
    instructions: str


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict


# ---------------------------------------------------------------------------
# Reset / Step request & response
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[Literal["easy", "medium", "hard"]] = None
    seed: Optional[int] = None


class ResetResponse(BaseModel):
    observation: Observation
    task_id: str
    max_steps: int


class StepRequest(BaseModel):
    action: Action


# ---------------------------------------------------------------------------
# WebSocket message envelopes
# ---------------------------------------------------------------------------

class WSMessage(BaseModel):
    type: Literal["reset", "step", "state", "ping"]
    payload: Optional[Dict] = None


class WSResponse(BaseModel):
    type: Literal["reset_ack", "step_result", "state_result", "pong", "error"]
    payload: Dict
