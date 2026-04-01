"""
Typed Pydantic models for CustomerSupportEnv (OpenEnv spec).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ── Enumerations ──────────────────────────────────────────────────────────────

class TicketStatus(str, Enum):
    IDLE = "idle"
    OPEN = "open"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    TIMEOUT = "timeout"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Category(str, Enum):
    AUTH = "auth"
    BILLING = "billing"
    FULFILLMENT = "fulfillment"
    BUG = "bug"
    SALES = "sales"
    GENERAL = "general"


class ActionType(str, Enum):
    SEARCH_KB = "search_kb"
    EMPATHIZE = "empathize"
    ASK_CLARIFY = "ask_clarify"
    OFFER_SOLUTION = "offer_solution"
    ESCALATE = "escalate"
    RESOLVE = "resolve"
    SEND_MESSAGE = "send_message"


# ── Core Typed Models ─────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str  # "customer" | "agent" | "system"
    text: str
    turn: int = 0


class Observation(BaseModel):
    """Full typed observation returned by reset() and step()."""
    ticket_id: Optional[str] = None
    task_id: str = "task_1"
    status: TicketStatus = TicketStatus.IDLE
    sentiment: Optional[Sentiment] = None
    priority: Optional[Priority] = None
    category: Optional[Category] = None
    turn: int = 0
    max_turns: int = 10
    history: List[Message] = Field(default_factory=list)
    kb_results: List[str] = Field(default_factory=list)
    kb_searched: bool = False
    empathized: bool = False
    clarified: bool = False
    solution_offered: bool = False
    escalated: bool = False
    cumulative_reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class Action(BaseModel):
    """Typed action submitted by the agent via step()."""
    action_type: ActionType
    payload: Optional[str] = None  # free-text for send_message / offer_solution

    class Config:
        use_enum_values = True


class Reward(BaseModel):
    """Typed reward with decomposed components."""
    total: float
    process_score: float = 0.0   # correct action sequencing
    quality_score: float = 0.0   # solution quality / empathy
    efficiency_score: float = 0.0  # steps taken vs optimal
    csat_score: float = 0.0      # synthetic customer satisfaction (0–1)
    penalties: float = 0.0
    reason: str = ""


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    """Defines one graded task within the environment."""
    task_id: str
    name: str
    description: str
    difficulty: str           # easy | medium | hard
    ticket_id: str
    success_criteria: List[str]
    max_turns: int
    optimal_actions: List[str]


class GraderResult(BaseModel):
    task_id: str
    score: float              # 0.0 – 1.0
    breakdown: Dict[str, float]
    passed: bool
    reason: str
