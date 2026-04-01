"""
CustomerSupportEnv — Core environment implementing the OpenEnv spec.

step(action)  → StepResult(observation, reward, done, info)
reset()       → Observation
state()       → Observation
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    Action, ActionType, Category, Message, Observation,
    Priority, Reward, Sentiment, StepResult, TaskSpec, TicketStatus
)
from env.tickets import TICKETS, get_ticket


# ── Reward constants ──────────────────────────────────────────────────────────

R_SEARCH_KB        =  2.0
R_EMPATHIZE        =  1.0
R_ASK_CLARIFY      =  1.0
R_OFFER_SOLUTION   =  3.0
R_RESOLVE_GOOD     =  5.0
R_RESOLVE_BAD      = -3.0
R_ESCALATE         = -1.0
R_DUPLICATE_ACTION = -1.0
R_SKIP_KB_PENALTY  = -1.0
R_TIMEOUT          = -2.0

CSAT_WEIGHTS = {
    "empathized": 0.3,
    "kb_searched": 0.3,
    "solution_offered": 0.4,
}

# Optimal trajectory (used for efficiency scoring)
OPTIMAL_STEPS = 4  # search_kb, empathize, offer_solution, resolve


# ── Task definitions ──────────────────────────────────────────────────────────

TASKS: Dict[str, TaskSpec] = {
    "task_1": TaskSpec(
        task_id="task_1",
        name="Resolve a Standard Auth Ticket",
        description=(
            "Handle a frustrated customer locked out of their account. "
            "The agent must search the knowledge base, acknowledge the "
            "customer's frustration, offer a concrete solution, and resolve the ticket. "
            "EASY: single-step fix, KB articles directly address the issue."
        ),
        difficulty="easy",
        ticket_id="TKT-001",
        success_criteria=[
            "search_kb called before offer_solution",
            "empathize called at least once",
            "offer_solution payload mentions unlock or reset",
            "resolve called to close episode"
        ],
        max_turns=8,
        optimal_actions=["search_kb", "empathize", "offer_solution", "resolve"]
    ),
    "task_2": TaskSpec(
        task_id="task_2",
        name="Handle a Multi-Step Billing Dispute",
        description=(
            "Resolve a billing discrepancy for a customer who was overcharged after "
            "a plan downgrade. The agent must clarify details, check the KB, diagnose "
            "the root cause, provide a specific dollar credit, and confirm the fix. "
            "MEDIUM: requires clarification before diagnosis; generic solutions penalised."
        ),
        difficulty="medium",
        ticket_id="TKT-003",
        success_criteria=[
            "ask_clarify called at least once",
            "search_kb called",
            "offer_solution mentions credit or refund amount",
            "resolve called"
        ],
        max_turns=10,
        optimal_actions=["search_kb", "ask_clarify", "empathize", "offer_solution", "resolve"]
    ),
    "task_3": TaskSpec(
        task_id="task_3",
        name="Triage a Critical Time-Sensitive Bug Report",
        description=(
            "An enterprise customer has a compliance deadline tomorrow and a data export "
            "stuck at 12% for 6 hours. The agent must quickly diagnose the issue, "
            "deploy an immediate workaround (priority queue), offer a backup strategy "
            "(partial export), and close with a monitoring commitment. "
            "HARD: time pressure, two-part solution required, escalation penalised, "
            "generic solutions score low."
        ),
        difficulty="hard",
        ticket_id="TKT-006",
        success_criteria=[
            "search_kb called",
            "offer_solution mentions priority queue AND partial export",
            "solution demonstrates urgency awareness",
            "resolve called without escalation"
        ],
        max_turns=8,
        optimal_actions=["search_kb", "empathize", "ask_clarify", "offer_solution", "resolve"]
    )
}


# ── Environment ───────────────────────────────────────────────────────────────

class CustomerSupportEnv:
    """
    OpenEnv-compatible customer support RL environment.

    Usage:
        env = CustomerSupportEnv(task_id="task_1")
        obs = env.reset()
        result = env.step(Action(action_type="search_kb"))
        current = env.state()
    """

    VERSION = "1.0.0"

    def __init__(self, task_id: str = "task_1", seed: Optional[int] = None):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}")
        self.task_id = task_id
        self.task = TASKS[task_id]
        self._seed = seed
        self._rng = random.Random(seed)
        self._obs: Observation = self._make_idle_obs()

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        ticket_data = get_ticket(self.task.ticket_id)
        history = [
            Message(role=m["role"], text=m["text"], turn=m.get("turn", 0))
            for m in ticket_data["history"]
        ]
        self._obs = Observation(
            ticket_id=self.task.ticket_id,
            task_id=self.task_id,
            status=TicketStatus.OPEN,
            sentiment=ticket_data["sentiment"],
            priority=ticket_data["priority"],
            category=ticket_data["category"],
            turn=0,
            max_turns=self.task.max_turns,
            history=history,
            kb_results=[],
            kb_searched=False,
            empathized=False,
            clarified=False,
            solution_offered=False,
            escalated=False,
            cumulative_reward=0.0,
            done=False,
            info={"task_name": self.task.name, "difficulty": self.task.difficulty}
        )
        return self._obs

    def step(self, action: Action) -> StepResult:
        """
        Advance the environment by one step.
        Returns StepResult(observation, reward, done, info).
        """
        if self._obs.status == TicketStatus.IDLE:
            raise RuntimeError("Call reset() before step().")
        if self._obs.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        obs = self._obs
        ticket = get_ticket(obs.ticket_id)
        action_type = ActionType(action.action_type)

        step_reward, reason, penalty = 0.0, "", 0.0
        done = False
        info: Dict[str, Any] = {}

        obs.turn += 1

        # ── Dispatch action ────────────────────────────────────────────────

        if action_type == ActionType.SEARCH_KB:
            if obs.kb_searched:
                penalty = R_DUPLICATE_ACTION
                reason = "Duplicate search_kb — no new information."
            else:
                obs.kb_searched = True
                obs.kb_results = ticket["kb_articles"]
                step_reward = R_SEARCH_KB
                reason = f"Retrieved {len(obs.kb_results)} KB articles."

        elif action_type == ActionType.EMPATHIZE:
            if obs.empathized:
                reason = "Already empathized — no incremental reward."
            else:
                obs.empathized = True
                step_reward = R_EMPATHIZE
                reason = "Empathy acknowledged by customer."
            obs.history.append(Message(
                role="agent",
                text=self._rng.choice([
                    "I completely understand how frustrating this situation must be. Let me help you immediately.",
                    "I'm sorry you're going through this — that sounds really stressful. Let's fix it right away.",
                    "Thank you for reaching out. I can see why this is a concern and I want to resolve it for you."
                ]),
                turn=obs.turn
            ))
            obs.history.append(Message(
                role="customer",
                text=self._rng.choice(["I appreciate that, thank you.", "Ok, let's get this sorted.", "Thank you."]),
                turn=obs.turn
            ))

        elif action_type == ActionType.ASK_CLARIFY:
            if obs.clarified:
                reason = "Already clarified — no incremental reward."
            else:
                obs.clarified = True
                step_reward = R_ASK_CLARIFY
                reason = "Clarifying question logged."
            clarify_q = action.payload or "Could you share your account email and any relevant reference numbers?"
            obs.history.append(Message(role="agent", text=clarify_q, turn=obs.turn))
            obs.history.append(Message(
                role="customer",
                text=self._rng.choice([
                    "My account email is user@example.com. Order reference #482923.",
                    "Sure — account email user@example.com, invoice #8821.",
                    "My email is user@example.com. It started 3 days ago."
                ]),
                turn=obs.turn
            ))

        elif action_type == ActionType.OFFER_SOLUTION:
            if not obs.kb_searched:
                penalty = R_SKIP_KB_PENALTY
                reason = "Penalty: solution offered without consulting the knowledge base."
            solution_text = action.payload or ticket["canonical_solution"]
            quality = self._score_solution(solution_text, ticket)
            obs.solution_offered = True
            step_reward = R_OFFER_SOLUTION * quality
            reason = f"Solution offered. Quality score: {quality:.2f}."
            info["solution_quality"] = quality
            obs.history.append(Message(role="agent", text=solution_text, turn=obs.turn))
            obs.history.append(Message(
                role="customer",
                text=self._rng.choice(ticket["customer_followups"]),
                turn=obs.turn
            ))

        elif action_type == ActionType.ESCALATE:
            if obs.escalated:
                penalty = R_DUPLICATE_ACTION * 2
                reason = "Double escalation penalty."
            else:
                obs.escalated = True
                penalty = R_ESCALATE
                reason = "Escalated to tier-2. In-tier resolution preferred."
            obs.history.append(Message(
                role="system",
                text="Ticket escalated to tier-2 specialist team.",
                turn=obs.turn
            ))

        elif action_type == ActionType.RESOLVE:
            done = True
            obs.status = TicketStatus.RESOLVED if not obs.escalated else TicketStatus.ESCALATED
            if obs.solution_offered or obs.escalated:
                csat = self._compute_csat(obs)
                step_reward = R_RESOLVE_GOOD + csat * 2.0
                reason = f"Resolved. CSAT: {csat:.2f}/1.0"
                info["csat"] = csat
            else:
                step_reward = R_RESOLVE_BAD
                reason = "Penalty: resolved without offering a solution."
            obs.history.append(Message(
                role="agent",
                text="Thank you for your patience. I'm marking this ticket as resolved. Please don't hesitate to reach out if you need further help.",
                turn=obs.turn
            ))

        elif action_type == ActionType.SEND_MESSAGE:
            # Free-form message — small reward for engagement
            msg = action.payload or "I'm looking into this for you."
            obs.history.append(Message(role="agent", text=msg, turn=obs.turn))
            step_reward = 0.5
            reason = "Message sent."

        # ── Timeout check ─────────────────────────────────────────────────

        if obs.turn >= obs.max_turns and not done:
            penalty += R_TIMEOUT
            done = True
            obs.status = TicketStatus.TIMEOUT
            reason += " | Episode timed out."

        # ── Build reward ──────────────────────────────────────────────────

        net = step_reward + penalty
        efficiency = max(0.0, 1.0 - max(0, obs.turn - OPTIMAL_STEPS) * 0.1)
        process = min(1.0, (
            (0.25 if obs.kb_searched else 0) +
            (0.25 if obs.empathized else 0) +
            (0.25 if obs.solution_offered else 0) +
            (0.25 if done and obs.status == TicketStatus.RESOLVED else 0)
        ))
        reward = Reward(
            total=round(net, 3),
            process_score=round(process, 3),
            quality_score=round(info.get("solution_quality", 0.0), 3),
            efficiency_score=round(efficiency, 3),
            csat_score=round(info.get("csat", 0.0), 3),
            penalties=round(penalty, 3),
            reason=reason
        )

        obs.cumulative_reward = round(obs.cumulative_reward + net, 3)
        obs.done = done
        info["turn"] = obs.turn
        info["cumulative_reward"] = obs.cumulative_reward
        obs.info = info
        self._obs = obs

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> Observation:
        """Return current observation without advancing the environment."""
        return self._obs

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_idle_obs(self) -> Observation:
        return Observation(task_id=self.task_id)

    def _score_solution(self, solution_text: str, ticket: dict) -> float:
        """Score solution quality against expected keywords (0.0–1.0)."""
        text_lower = solution_text.lower()
        keywords = ticket.get("solution_keywords", [])
        if not keywords:
            return 0.5
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        return min(1.0, hits / max(1, len(keywords)))

    def _compute_csat(self, obs: Observation) -> float:
        """Synthetic CSAT score (0.0–1.0) based on interaction quality."""
        score = 0.0
        if obs.empathized:
            score += CSAT_WEIGHTS["empathized"]
        if obs.kb_searched:
            score += CSAT_WEIGHTS["kb_searched"]
        if obs.solution_offered:
            score += CSAT_WEIGHTS["solution_offered"]
        return round(score, 3)

    @staticmethod
    def list_tasks() -> List[str]:
        return list(TASKS.keys())

    @staticmethod
    def get_task_spec(task_id: str) -> TaskSpec:
        return TASKS[task_id]
