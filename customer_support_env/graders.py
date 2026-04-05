"""
Pure grading functions for all 3 tasks.
All functions are deterministic given the same inputs.
No I/O, no randomness, no external calls.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .models import Action, PolicyResult
from .tasks import GroundTruth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keywords_present(text: Optional[str], keywords: List[str]) -> bool:
    """Return True if ALL keywords appear in text (case-insensitive)."""
    if not text:
        return False
    lower = text.lower()
    return all(kw.lower() in lower for kw in keywords)


def _last_action_of_type(actions: List[Action], action_type: str) -> Optional[Action]:
    """Return the most recent action of the given type, or None."""
    for a in reversed(actions):
        if a.action_type == action_type:
            return a
    return None


def _action_index(actions: List[Action], action_type: str) -> int:
    """Return the index of the FIRST action of given type, or -1 if not found."""
    for i, a in enumerate(actions):
        if a.action_type == action_type:
            return i
    return -1


# ---------------------------------------------------------------------------
# Easy task grader — max score 1.0
# ---------------------------------------------------------------------------

def grade_easy(
    actions_taken: List[Action],
    ground_truth: GroundTruth,
    step_count: int,
    _last_policy_result: Optional[PolicyResult] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Sub-rewards:
      classify_correct   0.30
      priority_correct   0.20
      response_sent      0.20
      response_quality   0.20
      efficiency_bonus   0.10
    Total max: 1.00
    """
    breakdown: Dict[str, float] = {
        "classify_correct": 0.0,
        "priority_correct": 0.0,
        "response_sent": 0.0,
        "response_quality": 0.0,
        "efficiency_bonus": 0.0,
    }

    classify_action = _last_action_of_type(actions_taken, "classify")
    if classify_action and classify_action.category == ground_truth.correct_category:
        breakdown["classify_correct"] = 0.30

    priority_action = _last_action_of_type(actions_taken, "set_priority")
    if priority_action and priority_action.priority == ground_truth.correct_priority:
        breakdown["priority_correct"] = 0.20

    respond_action = _last_action_of_type(actions_taken, "respond")
    if respond_action and respond_action.response_text:
        breakdown["response_sent"] = 0.20
        kws = ground_truth.required_response_keywords or []
        if kws and _keywords_present(respond_action.response_text, kws):
            breakdown["response_quality"] = 0.20
        elif not kws:
            # No keywords required — full quality score for any response
            breakdown["response_quality"] = 0.20

    if step_count <= 4:  # 4 required actions; bonus = no wasted steps
        breakdown["efficiency_bonus"] = 0.10

    total = round(sum(breakdown.values()), 4)
    return total, breakdown


# ---------------------------------------------------------------------------
# Medium task grader — max score 1.0
# ---------------------------------------------------------------------------

def grade_medium(
    actions_taken: List[Action],
    ground_truth: GroundTruth,
    step_count: int,
    _last_policy_result: Optional[PolicyResult] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Sub-rewards:
      triage_submitted    0.10
      category_accuracy   0.40 * (correct / total)
      priority_accuracy   0.40 * (correct / total)
      efficiency_bonus    0.10
    Total max: 1.00
    """
    breakdown: Dict[str, float] = {
        "triage_submitted": 0.0,
        "category_accuracy": 0.0,
        "priority_accuracy": 0.0,
        "efficiency_bonus": 0.0,
    }

    # Only the LAST bulk_triage is evaluated (allows corrections)
    triage_action = _last_action_of_type(actions_taken, "bulk_triage")
    if triage_action is None or triage_action.triage_list is None:
        return 0.0, breakdown

    breakdown["triage_submitted"] = 0.10

    gt_by_id = {item.ticket_id: item for item in (ground_truth.correct_triage or [])}
    submitted = {item.ticket_id: item for item in triage_action.triage_list}
    total = len(gt_by_id)

    if total == 0:
        return breakdown["triage_submitted"], breakdown

    cat_correct = 0
    pri_correct = 0
    for tid, gt_item in gt_by_id.items():
        sub_item = submitted.get(tid)
        if sub_item is None:
            continue
        if sub_item.category == gt_item.category:
            cat_correct += 1
        if sub_item.priority == gt_item.priority:
            pri_correct += 1

    breakdown["category_accuracy"] = round(0.40 * (cat_correct / total), 4)
    breakdown["priority_accuracy"] = round(0.40 * (pri_correct / total), 4)

    if step_count <= 3:
        breakdown["efficiency_bonus"] = 0.10

    total_score = round(sum(breakdown.values()), 4)
    return total_score, breakdown


# ---------------------------------------------------------------------------
# Hard task grader — max score 1.0
# ---------------------------------------------------------------------------

def grade_hard(
    actions_taken: List[Action],
    ground_truth: GroundTruth,
    _step_count: int,
    last_policy_result: Optional[PolicyResult] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Sub-rewards:
      classify_correct        0.10
      priority_correct        0.10
      policy_looked_up        0.10
      correct_policy_found    0.15
      escalated               0.10
      correct_escalation_tier 0.15
      respond_quality         0.15  ← must mention compensation/timeline to customer
      resolved                0.05
      resolution_quality      0.10  ← resolution note must contain all required keywords
    Total max: 1.00

    Difficulty: the hard task requires ALL steps to be done correctly AND the
    response + resolution must contain domain-specific language. Frontier models
    that skip steps or write generic notes will lose significant points.
    """
    breakdown: Dict[str, float] = {
        "classify_correct": 0.0,
        "priority_correct": 0.0,
        "policy_looked_up": 0.0,
        "correct_policy_found": 0.0,
        "escalated": 0.0,
        "correct_escalation_tier": 0.0,
        "respond_quality": 0.0,
        "resolved": 0.0,
        "resolution_quality": 0.0,
    }

    classify_action = _last_action_of_type(actions_taken, "classify")
    if classify_action and classify_action.category == ground_truth.correct_category:
        breakdown["classify_correct"] = 0.10

    priority_action = _last_action_of_type(actions_taken, "set_priority")
    if priority_action and priority_action.priority == ground_truth.correct_priority:
        breakdown["priority_correct"] = 0.10

    # Order-aware policy check: lookup_policy must occur BEFORE escalate
    lookup_idx = _action_index(actions_taken, "lookup_policy")
    escalate_idx = _action_index(actions_taken, "escalate")

    if lookup_idx >= 0:
        breakdown["policy_looked_up"] = 0.10
        # Correct policy must be found AND lookup must happen before escalation
        lookup_before_escalate = (escalate_idx < 0 or lookup_idx < escalate_idx)
        if (
            last_policy_result is not None
            and last_policy_result.matched_policy_id == ground_truth.required_policy_id
            and lookup_before_escalate
        ):
            breakdown["correct_policy_found"] = 0.15

    escalate_action = _last_action_of_type(actions_taken, "escalate")
    if escalate_action:
        breakdown["escalated"] = 0.10
        if escalate_action.escalation_tier == ground_truth.correct_escalation_tier:
            breakdown["correct_escalation_tier"] = 0.15

    # Respond quality: must mention SLA/timeline/credit from policy content
    # Generic "we escalated" is NOT enough — agent must read and apply policy
    respond_action = _last_action_of_type(actions_taken, "respond")
    if respond_action and respond_action.response_text:
        respond_kws = ground_truth.required_response_keywords or ["escalat", "investigat"]
        if _keywords_present(respond_action.response_text, respond_kws):
            breakdown["respond_quality"] = 0.15

    resolve_action = _last_action_of_type(actions_taken, "resolve")
    if resolve_action and resolve_action.resolution_note:
        breakdown["resolved"] = 0.05
        # Resolution note must reference policy-derived terms (not just generic notes)
        kws = ground_truth.correct_resolution_note_keywords or []
        if kws and _keywords_present(resolve_action.resolution_note, kws):
            breakdown["resolution_quality"] = 0.10
        elif not kws:
            breakdown["resolution_quality"] = 0.10

    total = round(sum(breakdown.values()), 4)
    return total, breakdown


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

GRADER_MAP = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade(
    task_id: str,
    actions_taken: List[Action],
    ground_truth: GroundTruth,
    step_count: int,
    last_policy_result: Optional[PolicyResult] = None,
) -> Tuple[float, Dict[str, float]]:
    """Dispatch to the appropriate grader for the given task."""
    grader = GRADER_MAP.get(task_id)
    if grader is None:
        raise ValueError(f"No grader for task_id={task_id!r}")
    return grader(actions_taken, ground_truth, step_count, last_policy_result)
