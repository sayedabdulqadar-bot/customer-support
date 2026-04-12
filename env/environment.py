# ADVANCED ENTERPRISE INCIDENT ENVIRONMENT

from __future__ import annotations
import random
from typing import Dict, Any, Optional, List

from env.models import (
    Action, ActionType, Message, Observation,
    Reward, StepResult, TaskSpec, TicketStatus
)
from env.tickets import get_ticket


# =========================
# TASK DEFINITIONS
# =========================
TASKS: Dict[str, TaskSpec] = {
    "task_1": TaskSpec(
        task_id="task_1",
        name="Basic Auth Issue",
        description="Resolve login issue with empathy and KB usage",
        difficulty="easy",
        ticket_id="TKT-001",
        max_turns=8
    ),
    "task_2": TaskSpec(
        task_id="task_2",
        name="Billing SLA Case",
        description="Resolve billing with SLA and refund precision",
        difficulty="medium",
        ticket_id="TKT-003",
        max_turns=10
    ),
    "task_3": TaskSpec(
        task_id="task_3",
        name="Critical Enterprise Outage",
        description="Handle high severity incident with urgency",
        difficulty="hard",
        ticket_id="TKT-006",
        max_turns=8
    )
}


# =========================
# ENVIRONMENT
# =========================
class CustomerSupportEnv:

    def __init__(self, task_id="task_1", seed=None):
        self.task = TASKS[task_id]
        self._rng = random.Random(seed)
        self._obs: Observation = None

    def reset(self) -> Observation:
        ticket = get_ticket(self.task.ticket_id)

        # 🔥 ADVANCED CONTEXT
        self.sla_deadline = self._rng.choice([3, 5, 7])
        self.customer_tier = self._rng.choice(["free", "premium", "enterprise"])
        self.issue_severity = self._rng.choice(["low", "medium", "critical"])
        self.escalation_risk = self._rng.choice([0.2, 0.5, 0.8])
        self.hidden_failure_mode = self._rng.choice([True, False])

        history = [
            Message(role=m["role"], text=m["text"], turn=0)
            for m in ticket["history"]
        ]

        self._obs = Observation(
            ticket_id=self.task.ticket_id,
            task_id=self.task.task_id,
            status=TicketStatus.OPEN,
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
            info={
                "sla": self.sla_deadline,
                "tier": self.customer_tier,
                "severity": self.issue_severity,
                "risk": self.escalation_risk
            }
        )
        return self._obs

    def step(self, action: Action) -> StepResult:
        obs = self._obs
        action_type = action.action_type

        reward = 0.0
        done = False

        obs.turn += 1

        # =========================
        # ACTIONS
        # =========================
        if action_type == "search_kb":
            reward += 2 if not obs.kb_searched else -1
            obs.kb_searched = True

        elif action_type == "empathize":
            reward += 1 if not obs.empathized else 0
            obs.empathized = True

        elif action_type == "ask_clarify":
            reward += 1 if not obs.clarified else 0
            obs.clarified = True

        elif action_type == "offer_solution":
            reward += 3
            obs.solution_offered = True

        elif action_type == "resolve":
            done = True
            reward += 5 if obs.solution_offered else -3

        elif action_type == "escalate":
            obs.escalated = True
            reward -= 1

        # =========================
        # ADVANCED LOGIC
        # =========================

        # SLA pressure
        if obs.turn > self.sla_deadline:
            reward -= 1

        # severity boost
        if self.issue_severity == "critical":
            if action_type == "search_kb":
                reward += 0.5

        # enterprise expectations
        if self.customer_tier == "enterprise":
            if done and obs.turn <= self.sla_deadline:
                reward += 2
            elif done:
                reward -= 1

        # escalation risk
        if self.escalation_risk > 0.7 and not obs.empathized:
            reward -= 1

        # hidden failure mode (novel)
        if self.hidden_failure_mode and action_type == "offer_solution" and not obs.kb_searched:
            reward -= 2

        # auto escalation
        if self.escalation_risk > 0.7 and obs.turn > 3 and not obs.empathized:
            done = True
            obs.escalated = True
            reward -= 3

        # efficiency
        if done and obs.turn <= 4:
            reward += 1

        obs.done = done
        obs.cumulative_reward += reward

        return StepResult(
            observation=obs,
            reward=Reward(total=round(reward, 3)),
            done=done,
            info={}
        )
