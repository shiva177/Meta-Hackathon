"""
Unit tests for the Customer Support Ticket Resolution Environment.
Run with: pytest tests/test_env.py -v
"""

import pytest

from customer_support_env import (
    Action,
    CustomerSupportEnv,
    Observation,
    StepResult,
)
from customer_support_env.graders import grade_easy, grade_hard, grade_medium
from customer_support_env.models import TriageItem
from customer_support_env.policies import lookup_policy
from customer_support_env.tasks import GroundTruth, get_expected_priority, load_task


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return CustomerSupportEnv()


# ---------------------------------------------------------------------------
# Environment lifecycle tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_easy_returns_observation(self, env):
        obs = env.reset(task_id="easy", seed=42)
        assert isinstance(obs, Observation)
        assert obs.task_id == "easy"
        assert obs.step == 0
        assert len(obs.tickets) == 1

    def test_reset_medium_returns_5_tickets(self, env):
        obs = env.reset(task_id="medium", seed=42)
        assert obs.task_id == "medium"
        assert len(obs.tickets) == 5

    def test_reset_hard_returns_1_ticket(self, env):
        obs = env.reset(task_id="hard", seed=42)
        assert obs.task_id == "hard"
        assert len(obs.tickets) == 1

    def test_reset_clears_previous_state(self, env):
        env.reset(task_id="easy", seed=42)
        ticket_id = env._task_instance.tickets[0].ticket_id
        env.step(Action(action_type="classify", category="billing"))
        env.reset(task_id="easy", seed=99)
        assert env._current_step == 0
        assert env._actions_taken == []
        assert env._cumulative_reward == 0.0
        assert env._done is False

    def test_reset_reproducible_with_seed(self, env):
        obs1 = env.reset(task_id="easy", seed=42)
        tid1 = obs1.tickets[0].ticket_id
        obs2 = env.reset(task_id="easy", seed=42)
        tid2 = obs2.tickets[0].ticket_id
        assert tid1 == tid2

    def test_invalid_task_id_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="expert")


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_before_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(Action(action_type="classify", category="billing"))

    def test_valid_classify_step(self, env):
        env.reset(task_id="easy", seed=42)
        result = env.step(Action(action_type="classify", category="billing"))
        assert isinstance(result, StepResult)
        assert result.info.get("error") is None
        assert result.observation.step == 1
        assert result.observation.action_valid is True

    def test_invalid_classify_missing_category(self, env):
        env.reset(task_id="easy", seed=42)
        result = env.step(Action(action_type="classify"))
        assert result.info.get("error") is not None
        assert result.observation.action_valid is False

    def test_bulk_triage_invalid_on_easy_task(self, env):
        env.reset(task_id="easy", seed=42)
        result = env.step(
            Action(
                action_type="bulk_triage",
                triage_list=[TriageItem(ticket_id="TKT-001", category="billing", priority="high")],
            )
        )
        assert result.info.get("error") is not None

    def test_done_after_resolve_on_easy(self, env):
        obs = env.reset(task_id="easy", seed=42)
        ticket_id = obs.tickets[0].ticket_id
        env.step(Action(action_type="classify", category="billing"))
        env.step(Action(action_type="set_priority", priority="high"))
        env.step(Action(action_type="respond", response_text="We will refund your charge."))
        result = env.step(
            Action(
                action_type="resolve",
                resolved_ticket_id=ticket_id,
                resolution_note="Processed refund for duplicate charge.",
            )
        )
        assert result.done is True

    def test_done_after_bulk_triage_on_medium(self, env):
        obs = env.reset(task_id="medium", seed=42)
        triage_list = [
            TriageItem(ticket_id=t.ticket_id, category="general", priority="medium")
            for t in obs.tickets
        ]
        result = env.step(
            Action(action_type="bulk_triage", triage_list=triage_list)
        )
        assert result.done is True

    def test_reward_increases_with_correct_actions(self, env):
        env.reset(task_id="easy", seed=42)
        gt = env._task_instance.ground_truth

        r0 = env._cumulative_reward
        env.step(Action(action_type="classify", category=gt.correct_category))
        r1 = env._cumulative_reward
        assert r1 > r0

        env.step(Action(action_type="set_priority", priority=gt.correct_priority))
        r2 = env._cumulative_reward
        assert r2 > r1

    def test_reward_decreases_on_invalid_action(self, env):
        env.reset(task_id="easy", seed=42)
        env.step(Action(action_type="classify", category="billing"))
        r1 = env._cumulative_reward
        # Invalid action — empty response_text should trigger penalty
        env.step(Action(action_type="respond", response_text=""))
        r2 = env._cumulative_reward
        # Penalty of -0.05 applied, reward should be lower (floor 0.0)
        assert r2 < r1


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

class TestState:
    def test_state_before_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.state()

    def test_state_matches_last_observation(self, env):
        env.reset(task_id="easy", seed=42)
        env.step(Action(action_type="classify", category="billing"))
        obs_from_step = env._build_observation()
        obs_from_state = env.state()
        assert obs_from_state.step == obs_from_step.step
        assert obs_from_state.task_id == obs_from_step.task_id

    def test_state_does_not_advance_step(self, env):
        env.reset(task_id="easy", seed=42)
        env.state()
        env.state()
        assert env._current_step == 0


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------

class TestGraders:
    def _make_gt_easy(self):
        return GroundTruth(
            correct_category="billing",
            correct_priority="high",
            required_response_keywords=["refund", "charge"],
        )

    def test_easy_perfect_score(self):
        gt = self._make_gt_easy()
        actions = [
            Action(action_type="classify", category="billing"),
            Action(action_type="set_priority", priority="high"),
            Action(
                action_type="respond",
                response_text="We will process your refund for the duplicate charge immediately.",
            ),
            Action(
                action_type="resolve",
                resolved_ticket_id="TKT-001",
                resolution_note="Refund processed.",
            ),
        ]
        score, breakdown = grade_easy(actions, gt, step_count=4)
        # classify=0.3, priority=0.2, response_sent=0.2, response_quality=0.2, efficiency=0.1
        assert score == pytest.approx(1.0, abs=0.01)
        assert breakdown["classify_correct"] == 0.30
        assert breakdown["priority_correct"] == 0.20
        assert breakdown["response_sent"] == 0.20
        assert breakdown["response_quality"] == 0.20
        assert breakdown["efficiency_bonus"] == 0.10

    def test_easy_wrong_category_no_classify_reward(self):
        gt = self._make_gt_easy()
        actions = [Action(action_type="classify", category="technical")]
        score, breakdown = grade_easy(actions, gt, step_count=1)
        assert breakdown["classify_correct"] == 0.0

    def test_easy_no_response_quality_without_keywords(self):
        gt = self._make_gt_easy()
        actions = [
            Action(action_type="classify", category="billing"),
            Action(action_type="set_priority", priority="high"),
            Action(action_type="respond", response_text="We will look into your issue."),
        ]
        score, breakdown = grade_easy(actions, gt, step_count=3)
        assert breakdown["response_sent"] == 0.20
        assert breakdown["response_quality"] == 0.0   # keywords missing

    def test_medium_perfect_triage(self):
        correct = [
            TriageItem(ticket_id="TKT-001", category="billing", priority="high"),
            TriageItem(ticket_id="TKT-002", category="technical", priority="critical"),
            TriageItem(ticket_id="TKT-003", category="abuse", priority="critical"),
        ]
        gt = GroundTruth(correct_triage=correct)
        actions = [
            Action(
                action_type="bulk_triage",
                triage_list=correct,
            )
        ]
        score, breakdown = grade_medium(actions, gt, step_count=1)
        assert breakdown["triage_submitted"] == 0.10
        assert breakdown["category_accuracy"] == pytest.approx(0.40, abs=0.01)
        assert breakdown["priority_accuracy"] == pytest.approx(0.40, abs=0.01)
        assert breakdown["efficiency_bonus"] == 0.10

    def test_medium_partial_triage(self):
        correct = [
            TriageItem(ticket_id="TKT-001", category="billing", priority="high"),
            TriageItem(ticket_id="TKT-002", category="technical", priority="critical"),
        ]
        gt = GroundTruth(correct_triage=correct)
        # Only category of TKT-001 is correct
        submitted = [
            TriageItem(ticket_id="TKT-001", category="billing", priority="low"),  # wrong priority
            TriageItem(ticket_id="TKT-002", category="general", priority="critical"),  # wrong category
        ]
        actions = [Action(action_type="bulk_triage", triage_list=submitted)]
        score, breakdown = grade_medium(actions, gt, step_count=2)
        assert breakdown["category_accuracy"] == pytest.approx(0.20, abs=0.01)  # 1/2 * 0.40
        assert breakdown["priority_accuracy"] == pytest.approx(0.20, abs=0.01)  # 1/2 * 0.40

    def test_hard_full_workflow(self):
        from customer_support_env.models import PolicyResult
        gt = GroundTruth(
            correct_category="billing",
            correct_priority="critical",
            escalation_required=True,
            correct_escalation_tier="tier3",
            required_policy_id="POL-BILLING-002",
            required_response_keywords=["escalat", "investigat"],
            correct_resolution_note_keywords=["refund", "escalated", "account manager"],
        )
        policy_result = PolicyResult(
            query="enterprise billing invoice dispute",
            matched_policy_id="POL-BILLING-002",
            policy_title="Enterprise Contract and Invoice Disputes",
            confidence=0.8,
        )
        actions = [
            Action(action_type="classify", category="billing"),
            Action(action_type="set_priority", priority="critical"),
            Action(action_type="lookup_policy", policy_query="enterprise billing invoice dispute"),
            Action(
                action_type="escalate",
                escalation_tier="tier3",
                escalation_reason="Enterprise billing dispute exceeds $1000",
            ),
            Action(
                action_type="respond",
                response_text="We have escalated your case and are investigating the billing discrepancy.",
            ),
            Action(
                action_type="resolve",
                resolved_ticket_id="TKT-009",
                resolution_note="Ticket refund processed. Escalated to tier3. Account manager assigned.",
            ),
        ]
        score, breakdown = grade_hard(actions, gt, 6, last_policy_result=policy_result)
        assert score == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Policy lookup tests
# ---------------------------------------------------------------------------

class TestPolicyLookup:
    def test_billing_query_matches_billing_policy(self):
        result = lookup_policy("refund duplicate charge billing")
        assert result.matched_policy_id is not None
        assert "BILLING" in result.matched_policy_id
        assert result.confidence > 0

    def test_no_match_returns_null(self):
        result = lookup_policy("xyzzy completely irrelevant query zzz")
        assert result.matched_policy_id is None
        assert result.confidence == 0.0

    def test_escalation_query_matches_escalation_policy(self):
        result = lookup_policy("escalate manager supervisor tier3 critical")
        assert result.matched_policy_id == "POL-ESCALATION-001"

    def test_deterministic_same_query(self):
        r1 = lookup_policy("enterprise invoice billing dispute")
        r2 = lookup_policy("enterprise invoice billing dispute")
        assert r1.matched_policy_id == r2.matched_policy_id
        assert r1.confidence == r2.confidence


# ---------------------------------------------------------------------------
# Priority matrix tests
# ---------------------------------------------------------------------------

class TestPriorityMatrix:
    def test_enterprise_billing_is_critical(self):
        assert get_expected_priority("enterprise", "billing") == "critical"

    def test_abuse_always_critical(self):
        assert get_expected_priority("free", "abuse") == "critical"
        assert get_expected_priority("pro", "abuse") == "critical"
        assert get_expected_priority("enterprise", "abuse") == "critical"

    def test_free_technical_is_low(self):
        assert get_expected_priority("free", "technical") == "low"

    def test_pro_billing_is_high(self):
        assert get_expected_priority("pro", "billing") == "high"

    def test_feature_request_always_low(self):
        for tier in ("free", "pro", "enterprise"):
            assert get_expected_priority(tier, "feature_request") == "low"


# ---------------------------------------------------------------------------
# Task loading tests
# ---------------------------------------------------------------------------

class TestTaskLoading:
    def test_load_easy(self):
        task = load_task("easy", seed=42)
        assert task.spec.task_id == "easy"
        assert len(task.tickets) == 1
        assert task.ground_truth.correct_category is not None
        assert task.ground_truth.correct_priority is not None

    def test_load_medium(self):
        task = load_task("medium", seed=42)
        assert task.spec.task_id == "medium"
        assert len(task.tickets) == 5
        assert task.ground_truth.correct_triage is not None
        assert len(task.ground_truth.correct_triage) == 5

    def test_load_hard(self):
        task = load_task("hard", seed=42)
        assert task.spec.task_id == "hard"
        assert len(task.tickets) == 1
        assert task.ground_truth.correct_escalation_tier == "tier3"
        assert task.ground_truth.escalation_required is True

    def test_reproducible_seed(self):
        t1 = load_task("medium", seed=100)
        t2 = load_task("medium", seed=100)
        ids1 = [t.ticket_id for t in t1.tickets]
        ids2 = [t.ticket_id for t in t2.tickets]
        assert ids1 == ids2

    def test_different_seeds_may_differ(self):
        t1 = load_task("medium", seed=1)
        t2 = load_task("medium", seed=999)
        ids1 = sorted(t.ticket_id for t in t1.tickets)
        ids2 = sorted(t.ticket_id for t in t2.tickets)
        # Not guaranteed to differ but highly likely with different seeds
        # At minimum both should have 5 tickets
        assert len(ids1) == 5
        assert len(ids2) == 5
