"""
customer_support_env — OpenEnv-compatible customer support ticket resolution environment.
"""

from .env import CustomerSupportEnv
from .models import (
    Action,
    Observation,
    PolicyResult,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResult,
    Ticket,
    TriageItem,
    WSMessage,
    WSResponse,
)

__version__ = "1.0.0"

__all__ = [
    "CustomerSupportEnv",
    "Observation",
    "Action",
    "StepResult",
    "Ticket",
    "TriageItem",
    "PolicyResult",
    "ResetRequest",
    "ResetResponse",
    "StepRequest",
    "WSMessage",
    "WSResponse",
]
