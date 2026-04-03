"""
CustomerSupportEnv — core environment class.
Stateful, holds one episode at a time. Called by server.py.
"""

from __future__ import annotations

import os
from typing import List, Optional

from .graders import grade
from .models import Action, Observation, PolicyResult, StepResult
from .policies import lookup_policy
from .tasks import TaskInstance, load_task


class CustomerSupportEnv:
    """
    OpenEnv-compatible environment for customer support ticket resolution.

    Usage:
        env = CustomerSupportEnv()
        obs = env.reset(task_id="easy", seed=42)
        result = env.step(action)
        obs = env.state()
    """

    def __init__(self) -> None:
        self._task_instance: Optional[TaskInstance] = None
        self._current_step: int = 0
        self._actions_taken: List[Action] = []
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._policy_result: Optional[PolicyResult] = None
        self._escalation_queue: List[str] = []
        self._resolved_tickets: List[str] = []
        self._active_ticket_id: Optional[str] = None
        self._last_action_feedback: Optional[str] = None
        self._last_action_valid: Optional[bool] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Observation:
        """Reset environment and return initial observation."""
        # Task precedence: explicit arg > env var > default "easy"
        if task_id is None:
            task_id = os.environ.get("TASK_ID", "easy")

        self._task_instance = load_task(task_id, seed=seed)
        self._current_step = 0
        self._actions_taken = []
        self._cumulative_reward = 0.0
        self._done = False
        self._policy_result = None
        self._escalation_queue = []
        self._resolved_tickets = []
        self._last_action_feedback = None
        self._last_action_valid = None

        # For hard task, set active ticket to the first ticket
        if task_id == "hard" and self._task_instance.tickets:
            self._active_ticket_id = self._task_instance.tickets[0].ticket_id
        else:
            self._active_ticket_id = None

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """Execute one action and return the result."""
        if self._task_instance is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._done:
            obs = self._build_observation()
            return StepResult(
                observation=obs,
                reward=self._cumulative_reward,
                done=True,
                info={"step_reward": 0.0, "breakdown": {}, "error": "Episode already done."},
            )

        # Validate action
        error_msg = self._validate_action(action)
        if error_msg:
            self._last_action_feedback = f"Invalid action: {error_msg}"
            self._last_action_valid = False
            self._current_step += 1
            obs = self._build_observation()
            return StepResult(
                observation=obs,
                reward=self._cumulative_reward,
                done=self._is_done(),
                info={"step_reward": 0.0, "breakdown": {}, "error": error_msg},
            )

        # Apply action side-effects
        self._apply_action(action)

        # Append to history
        self._actions_taken.append(action)
        self._last_action_valid = True
        self._last_action_feedback = self._describe_action(action)
        self._current_step += 1

        # Grade
        new_reward, breakdown = grade(
            task_id=self._task_instance.spec.task_id,
            actions_taken=self._actions_taken,
            ground_truth=self._task_instance.ground_truth,
            step_count=self._current_step,
            last_policy_result=self._policy_result,
        )
        step_reward = round(new_reward - self._cumulative_reward, 4)
        self._cumulative_reward = new_reward

        self._done = self._is_done()
        obs = self._build_observation()

        return StepResult(
            observation=obs,
            reward=self._cumulative_reward,
            done=self._done,
            info={
                "step_reward": step_reward,
                "breakdown": breakdown,
                "error": None,
            },
        )

    def state(self) -> Observation:
        """Return current observation without advancing state."""
        if self._task_instance is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._build_observation()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_action(self, action: Action) -> Optional[str]:
        """Return error string if action is invalid, else None."""
        task_id = self._task_instance.spec.task_id  # type: ignore[union-attr]

        if action.action_type == "classify":
            if not action.category:
                return "classify requires 'category' field."

        elif action.action_type == "set_priority":
            if not action.priority:
                return "set_priority requires 'priority' field."

        elif action.action_type == "respond":
            if not action.response_text or not action.response_text.strip():
                return "respond requires non-empty 'response_text'."

        elif action.action_type == "escalate":
            if not action.escalation_tier:
                return "escalate requires 'escalation_tier'."
            if not action.escalation_reason or not action.escalation_reason.strip():
                return "escalate requires non-empty 'escalation_reason'."

        elif action.action_type == "lookup_policy":
            if not action.policy_query or not action.policy_query.strip():
                return "lookup_policy requires non-empty 'policy_query'."

        elif action.action_type == "resolve":
            if not action.resolved_ticket_id:
                return "resolve requires 'resolved_ticket_id'."
            if not action.resolution_note or not action.resolution_note.strip():
                return "resolve requires non-empty 'resolution_note'."

        elif action.action_type == "bulk_triage":
            if task_id != "medium":
                return "bulk_triage is only valid for the 'medium' task."
            if not action.triage_list:
                return "bulk_triage requires a non-empty 'triage_list'."
            expected_count = len(self._task_instance.tickets)  # type: ignore[union-attr]
            if len(action.triage_list) != expected_count:
                return (
                    f"bulk_triage triage_list must contain exactly {expected_count} items "
                    f"(one per ticket). Got {len(action.triage_list)}."
                )

        return None

    def _apply_action(self, action: Action) -> None:
        """Apply side-effects of a valid action."""
        if action.action_type == "lookup_policy":
            self._policy_result = lookup_policy(action.policy_query or "")

        elif action.action_type == "escalate":
            ticket_id = (
                action.resolved_ticket_id
                or self._active_ticket_id
                or (self._task_instance.tickets[0].ticket_id if self._task_instance and self._task_instance.tickets else "")
            )
            if ticket_id and ticket_id not in self._escalation_queue:
                self._escalation_queue.append(ticket_id)

        elif action.action_type == "resolve":
            tid = action.resolved_ticket_id or ""
            if tid and tid not in self._resolved_tickets:
                self._resolved_tickets.append(tid)

    def _describe_action(self, action: Action) -> str:
        """Human-readable feedback string for an action."""
        if action.action_type == "classify":
            return f"Ticket classified as: {action.category}."
        if action.action_type == "set_priority":
            return f"Priority set to: {action.priority}."
        if action.action_type == "respond":
            preview = (action.response_text or "")[:80]
            return f"Response sent: \"{preview}{'...' if len(action.response_text or '') > 80 else ''}\""
        if action.action_type == "escalate":
            return f"Ticket escalated to {action.escalation_tier}: {action.escalation_reason}."
        if action.action_type == "lookup_policy":
            if self._policy_result and self._policy_result.matched_policy_id:
                return (
                    f"Policy found: [{self._policy_result.matched_policy_id}] "
                    f"{self._policy_result.policy_title} "
                    f"(confidence={self._policy_result.confidence:.2f})."
                )
            return "Policy lookup returned no match for that query."
        if action.action_type == "resolve":
            return f"Ticket {action.resolved_ticket_id} resolved."
        if action.action_type == "bulk_triage":
            count = len(action.triage_list) if action.triage_list else 0
            return f"Bulk triage submitted for {count} tickets."
        return f"Action {action.action_type} executed."

    def _is_done(self) -> bool:
        """Check if the episode is complete."""
        if self._task_instance is None:
            return False

        task_id = self._task_instance.spec.task_id
        max_steps = self._task_instance.spec.max_steps

        if self._current_step >= max_steps:
            return True

        if task_id == "easy":
            return any(a.action_type == "resolve" for a in self._actions_taken)

        if task_id == "medium":
            return any(a.action_type == "bulk_triage" for a in self._actions_taken)

        if task_id == "hard":
            return bool(
                any(a.action_type == "resolve" for a in self._actions_taken)
                and self._resolved_tickets
            )

        return False

    def _build_observation(self) -> Observation:
        """Construct an Observation from current state."""
        assert self._task_instance is not None

        return Observation(
            step=self._current_step,
            task_id=self._task_instance.spec.task_id,
            tickets=self._task_instance.tickets,
            active_ticket_id=self._active_ticket_id,
            policy_result=self._policy_result,
            action_feedback=self._last_action_feedback,
            action_valid=self._last_action_valid,
            escalation_queue=list(self._escalation_queue),
            resolved_tickets=list(self._resolved_tickets),
            instructions=self._task_instance.spec.instructions,
        )
