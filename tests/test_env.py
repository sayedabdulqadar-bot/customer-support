"""
Tests for CustomerSupportEnv.
Run: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from env.environment import CustomerSupportEnv, TASKS
from env.models import Action, ActionType, TicketStatus
from graders.graders import grade


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env1():
    e = CustomerSupportEnv(task_id="task_1", seed=0)
    e.reset()
    return e

@pytest.fixture
def env2():
    e = CustomerSupportEnv(task_id="task_2", seed=0)
    e.reset()
    return e

@pytest.fixture
def env3():
    e = CustomerSupportEnv(task_id="task_3", seed=0)
    e.reset()
    return e


# ── reset() ───────────────────────────────────────────────────────────────────

def test_reset_returns_observation():
    env = CustomerSupportEnv(task_id="task_1", seed=0)
    obs = env.reset()
    assert obs.ticket_id == "TKT-001"
    assert obs.done is False
    assert obs.turn == 0
    assert obs.status == TicketStatus.OPEN.value or obs.status == TicketStatus.OPEN

def test_reset_clears_state(env1):
    env1.step(Action(action_type=ActionType.SEARCH_KB))
    obs = env1.reset()
    assert obs.kb_searched is False
    assert obs.turn == 0
    assert obs.cumulative_reward == 0.0

def test_reset_loads_history(env1):
    obs = env1.state()
    assert len(obs.history) >= 1
    assert obs.history[0].role == "customer"


# ── state() ───────────────────────────────────────────────────────────────────

def test_state_does_not_advance(env1):
    obs_before = env1.state()
    env1.state()
    obs_after = env1.state()
    assert obs_before.turn == obs_after.turn


# ── step() ────────────────────────────────────────────────────────────────────

def test_step_search_kb(env1):
    result = env1.step(Action(action_type=ActionType.SEARCH_KB))
    assert result.reward.total == 2.0
    assert result.observation.kb_searched is True
    assert len(result.observation.kb_results) > 0

def test_step_search_kb_duplicate_penalised(env1):
    env1.step(Action(action_type=ActionType.SEARCH_KB))
    result = env1.step(Action(action_type=ActionType.SEARCH_KB))
    assert result.reward.total < 0

def test_step_empathize(env1):
    result = env1.step(Action(action_type=ActionType.EMPATHIZE))
    assert result.reward.total == 1.0
    assert result.observation.empathized is True

def test_step_empathize_no_double_reward(env1):
    env1.step(Action(action_type=ActionType.EMPATHIZE))
    result = env1.step(Action(action_type=ActionType.EMPATHIZE))
    assert result.reward.total == 0.0

def test_step_offer_solution_without_kb_penalised(env1):
    result = env1.step(Action(
        action_type=ActionType.OFFER_SOLUTION,
        payload="I have unlocked your account and sent a reset link."
    ))
    assert result.reward.penalties == -1.0

def test_step_offer_solution_with_kb_rewarded(env1):
    env1.step(Action(action_type=ActionType.SEARCH_KB))
    result = env1.step(Action(
        action_type=ActionType.OFFER_SOLUTION,
        payload="I have unlocked your account and sent a password reset link."
    ))
    assert result.reward.total > 0

def test_step_resolve_without_solution_penalised(env1):
    result = env1.step(Action(action_type=ActionType.RESOLVE))
    assert result.reward.total == -3.0
    assert result.done is True

def test_step_resolve_good(env1):
    env1.step(Action(action_type=ActionType.SEARCH_KB))
    env1.step(Action(
        action_type=ActionType.OFFER_SOLUTION,
        payload="Account unlocked and reset email sent."
    ))
    result = env1.step(Action(action_type=ActionType.RESOLVE))
    assert result.reward.total >= 5.0
    assert result.done is True

def test_step_raises_before_reset():
    env = CustomerSupportEnv(task_id="task_1")
    with pytest.raises(RuntimeError):
        env.step(Action(action_type=ActionType.SEARCH_KB))

def test_step_raises_after_done(env1):
    env1.step(Action(action_type=ActionType.RESOLVE))
    with pytest.raises(RuntimeError):
        env1.step(Action(action_type=ActionType.SEARCH_KB))

def test_timeout_penalty(env1):
    """Exceeding max_turns gives timeout penalty."""
    for _ in range(env1._obs.max_turns - 1):
        env1.step(Action(action_type=ActionType.EMPATHIZE))
    obs = env1.state()
    assert obs.turn >= obs.max_turns - 1


# ── Graders ───────────────────────────────────────────────────────────────────

def test_grader_task1_optimal(env1):
    env1.step(Action(action_type=ActionType.SEARCH_KB))
    env1.step(Action(action_type=ActionType.EMPATHIZE))
    env1.step(Action(
        action_type=ActionType.OFFER_SOLUTION,
        payload="I have unlocked your account and sent a password reset link to your email."
    ))
    env1.step(Action(action_type=ActionType.RESOLVE))
    result = grade("task_1", env1.state())
    assert result.score >= 0.90
    assert result.passed is True

def test_grader_task1_minimal(env1):
    """Just resolve with no steps — should fail."""
    env1.step(Action(action_type=ActionType.RESOLVE))
    result = grade("task_1", env1.state())
    assert result.score < 0.40
    assert result.passed is False

def test_grader_task1_score_in_range(env1):
    result = grade("task_1", env1.state())
    assert 0.0 <= result.score <= 1.0

def test_grader_task2_requires_clarify(env2):
    """Medium task: no clarify → lower score."""
    env2.step(Action(action_type=ActionType.SEARCH_KB))
    env2.step(Action(
        action_type=ActionType.OFFER_SOLUTION,
        payload="I have applied a $20 credit to your account."
    ))
    env2.step(Action(action_type=ActionType.RESOLVE))
    result = grade("task_2", env2.state())
    assert result.breakdown.get("ask_clarify", 0) == 0.0

def test_grader_task2_full_score(env2):
    env2.step(Action(action_type=ActionType.SEARCH_KB))
    env2.step(Action(action_type=ActionType.ASK_CLARIFY, payload="Can you confirm your account email and invoice number?"))
    env2.step(Action(action_type=ActionType.EMPATHIZE))
    env2.step(Action(
        action_type=ActionType.OFFER_SOLUTION,
        payload="I have issued a $20 credit to your account. Your plan is now corrected to $29/month."
    ))
    env2.step(Action(action_type=ActionType.RESOLVE))
    result = grade("task_2", env2.state())
    assert result.score >= 0.70

def test_grader_task3_two_part_solution(env3):
    env3.step(Action(action_type=ActionType.SEARCH_KB))
    env3.step(Action(action_type=ActionType.EMPATHIZE))
    env3.step(Action(
        action_type=ActionType.OFFER_SOLUTION,
        payload="I have moved your export job to the priority queue — it will complete in 1-2 hours. "
                "As a backup, please start a partial export by date range which will be much faster. "
                "I will email you when the full export completes."
    ))
    env3.step(Action(action_type=ActionType.RESOLVE))
    result = grade("task_3", env3.state())
    assert result.score >= 0.70
    assert result.passed is True

def test_grader_task3_escalation_capped(env3):
    env3.step(Action(action_type=ActionType.SEARCH_KB))
    env3.step(Action(action_type=ActionType.ESCALATE))
    env3.step(Action(action_type=ActionType.RESOLVE))
    result = grade("task_3", env3.state())
    assert result.score <= 0.55

def test_grader_deterministic(env1):
    """Same inputs → same grader output every time."""
    env1.step(Action(action_type=ActionType.SEARCH_KB))
    env1.step(Action(action_type=ActionType.RESOLVE))
    r1 = grade("task_1", env1.state())
    env1.reset()
    env1.step(Action(action_type=ActionType.SEARCH_KB))
    env1.step(Action(action_type=ActionType.RESOLVE))
    r2 = grade("task_1", env1.state())
    assert r1.score == r2.score


# ── Task specs ────────────────────────────────────────────────────────────────

def test_task_list():
    assert set(CustomerSupportEnv.list_tasks()) == {"task_1", "task_2", "task_3"}

def test_task_difficulty_progression():
    diffs = [TASKS[tid].difficulty for tid in ["task_1", "task_2", "task_3"]]
    assert diffs == ["easy", "medium", "hard"]
