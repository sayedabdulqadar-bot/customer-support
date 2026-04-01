"""
Programmatic graders for CustomerSupportEnv tasks.

Each grader accepts a completed Observation and returns a GraderResult
with a score in [0.0, 1.0] and a detailed breakdown.

Graders are deterministic — same inputs always produce same outputs.
"""
from __future__ import annotations
from typing import Dict

from env.models import GraderResult, Observation, TicketStatus
from env.tickets import get_ticket


# ── Grader registry ───────────────────────────────────────────────────────────

def grade_task_1(obs: Observation) -> GraderResult:
    """
    Task 1 (EASY): Resolve a standard auth ticket.
    Scoring:
      - 0.30  kb_searched before offer_solution
      - 0.25  empathize called at least once
      - 0.25  offer_solution payload mentions unlock/reset keywords
      - 0.20  resolve called (status == RESOLVED)
    """
    ticket = get_ticket("TKT-001")
    breakdown: Dict[str, float] = {}

    # Check conversation history for evidence of each required action
    agent_turns = [m.text.lower() for m in obs.history if m.role == "agent"]
    all_agent_text = " ".join(agent_turns)

    # 1. KB searched
    kb_score = 0.30 if obs.kb_searched else 0.0
    breakdown["kb_searched"] = kb_score

    # 2. Empathy expressed
    empathy_score = 0.25 if obs.empathized else 0.0
    breakdown["empathized"] = empathy_score

    # 3. Solution quality — unlock/reset keywords
    solution_keywords = ticket["solution_keywords"]
    kw_hits = sum(1 for kw in solution_keywords if kw in all_agent_text)
    sol_score = 0.25 * min(1.0, kw_hits / max(1, len(solution_keywords)))
    breakdown["solution_quality"] = round(sol_score, 3)

    # 4. Resolved cleanly (not timeout, not just escalated)
    resolved = obs.status == TicketStatus.RESOLVED.value or obs.status == TicketStatus.RESOLVED
    resolve_score = 0.20 if resolved else 0.0
    breakdown["resolved"] = resolve_score

    total = sum(breakdown.values())
    passed = total >= 0.70

    return GraderResult(
        task_id="task_1",
        score=round(total, 3),
        breakdown=breakdown,
        passed=passed,
        reason=_build_reason(breakdown, passed)
    )


def grade_task_2(obs: Observation) -> GraderResult:
    """
    Task 2 (MEDIUM): Multi-step billing dispute.
    Scoring:
      - 0.20  ask_clarify called
      - 0.20  kb_searched
      - 0.30  offer_solution mentions a specific credit/refund (amount or keyword)
      - 0.15  empathize called
      - 0.15  resolve called
    """
    ticket = get_ticket("TKT-003")
    breakdown: Dict[str, float] = {}
    all_agent_text = " ".join(m.text.lower() for m in obs.history if m.role == "agent")

    # 1. Clarification step
    breakdown["ask_clarify"] = 0.20 if obs.clarified else 0.0

    # 2. KB searched
    breakdown["kb_searched"] = 0.20 if obs.kb_searched else 0.0

    # 3. Specific solution with $ amount or keywords
    solution_keywords = ticket["solution_keywords"]
    kw_hits = sum(1 for kw in solution_keywords if kw in all_agent_text)
    # Extra check: requires a numeric/specific value, not just generic words
    has_amount = any(x in all_agent_text for x in ["$20", "twenty", "20 credit", "credit of"])
    quality = min(1.0, kw_hits / max(1, len(solution_keywords)))
    if has_amount:
        quality = min(1.0, quality + 0.3)
    breakdown["solution_quality"] = round(0.30 * quality, 3)

    # 4. Empathy
    breakdown["empathized"] = 0.15 if obs.empathized else 0.0

    # 5. Resolved
    resolved = obs.status in (TicketStatus.RESOLVED.value, TicketStatus.RESOLVED)
    breakdown["resolved"] = 0.15 if resolved else 0.0

    total = sum(breakdown.values())
    passed = total >= 0.70

    return GraderResult(
        task_id="task_2",
        score=round(total, 3),
        breakdown=breakdown,
        passed=passed,
        reason=_build_reason(breakdown, passed)
    )


def grade_task_3(obs: Observation) -> GraderResult:
    """
    Task 3 (HARD): Critical time-sensitive bug — data export stuck.
    Scoring:
      - 0.20  kb_searched
      - 0.15  empathize called
      - 0.35  solution mentions BOTH priority queue AND partial export (two-part solution)
      - 0.15  NOT escalated (in-tier resolution required for full score)
      - 0.15  resolve called
    Bonus deduction: -0.10 if escalated (overrides the 0.15 no-escalation credit)
    """
    ticket = get_ticket("TKT-006")
    breakdown: Dict[str, float] = {}
    all_agent_text = " ".join(m.text.lower() for m in obs.history if m.role == "agent")

    # 1. KB searched
    breakdown["kb_searched"] = 0.20 if obs.kb_searched else 0.0

    # 2. Empathy
    breakdown["empathized"] = 0.15 if obs.empathized else 0.0

    # 3. Two-part solution: priority queue + partial export
    has_priority_queue = any(x in all_agent_text for x in ["priority queue", "priority export", "move your", "moved your"])
    has_partial = any(x in all_agent_text for x in ["partial", "date range", "by quarter", "partial export"])
    has_urgency = any(x in all_agent_text for x in ["deadline", "1-2 hour", "urgent", "compliance", "monitor", "email you"])

    sol_quality = 0.0
    if has_priority_queue and has_partial:
        sol_quality = 1.0
    elif has_priority_queue or has_partial:
        sol_quality = 0.5
    if has_urgency:
        sol_quality = min(1.0, sol_quality + 0.2)

    breakdown["solution_quality"] = round(0.35 * sol_quality, 3)

    # 4. No escalation
    breakdown["no_escalation"] = 0.0 if obs.escalated else 0.15

    # 5. Resolved
    resolved = obs.status in (TicketStatus.RESOLVED.value, TicketStatus.RESOLVED)
    breakdown["resolved"] = 0.15 if resolved else 0.0

    total = sum(breakdown.values())
    # Hard cap at 0.85 if escalated (escalation shows poor judgment on this task)
    if obs.escalated:
        total = min(total, 0.55)

    passed = total >= 0.70

    return GraderResult(
        task_id="task_3",
        score=round(total, 3),
        breakdown=breakdown,
        passed=passed,
        reason=_build_reason(breakdown, passed)
    )


GRADERS = {
    "task_1": grade_task_1,
    "task_2": grade_task_2,
    "task_3": grade_task_3,
}


def grade(task_id: str, obs: Observation) -> GraderResult:
    """Grade a completed observation for the given task."""
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task_id '{task_id}'. Valid: {list(GRADERS.keys())}")
    return GRADERS[task_id](obs)


def _build_reason(breakdown: Dict[str, float], passed: bool) -> str:
    hits = [k for k, v in breakdown.items() if v > 0]
    misses = [k for k, v in breakdown.items() if v == 0]
    status = "PASS" if passed else "FAIL"
    msg = f"[{status}] Score components present: {hits}."
    if misses:
        msg += f" Missing: {misses}."
    return msg
