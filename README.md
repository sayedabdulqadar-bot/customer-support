# CustomerSupportEnv

> An OpenEnv-compatible reinforcement learning environment for training and evaluating AI customer support agents.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0.0-blue)](openenv.yaml)
[![HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-yellow)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-ready-brightgreen)](Dockerfile)

---

## Overview

**CustomerSupportEnv** simulates a real-world Tier-1 customer support workflow. An agent handles inbound support tickets by searching a knowledge base, empathising with customers, asking clarifying questions, and delivering concrete solutions — all within a multi-turn conversation.

This environment is designed for:
- Training RL agents on real-world NLP tasks
- Benchmarking LLM-based tool-use and retrieval-augmented reasoning
- Evaluating customer satisfaction optimisation policies

---

## Quick Start

### Docker (recommended)
```bash
git clone https://huggingface.co/spaces/<your-username>/customer-support-env
cd customer-support-env
docker build -t customer-support-env .
docker run -p 7860:7860 customer-support-env
```

### Local
```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Run baseline inference
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
python inference.py
```

---

## Environment Description

Each **episode** = one customer support ticket. The agent takes a sequence of actions (turns) until it calls `resolve()` or exceeds `max_turns`.

### Real-world fidelity
- Tickets span 5 categories: **auth**, **billing**, **fulfillment**, **bug**, **sales**
- Customers have dynamic sentiment: **positive / neutral / frustrated / angry**
- Knowledge base retrieval is gated — agent must explicitly call `search_kb`
- Conversation history accumulates across turns, mirroring real support tooling
- CSAT (customer satisfaction) is a synthetic secondary objective

---

## OpenEnv API

### `POST /reset`
```json
{ "task_id": "task_1" }
```
Returns an `Observation`. Initialises a fresh episode.

### `POST /step`
```json
{ "task_id": "task_1", "action_type": "search_kb", "payload": null }
```
Returns a `StepResult` containing `observation`, `reward`, `done`, `info`.

### `GET /state?task_id=task_1`
Returns the current `Observation` without advancing the environment.

### `POST /grade`
```json
{ "task_id": "task_1" }
```
Returns a `GraderResult` with score (0.0–1.0), breakdown, and pass/fail.

### `GET /tasks`
Lists all task specs.

### `GET /health`
Returns `{"status": "ok"}`.

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | string | Ticket identifier (e.g. `TKT-001`) |
| `task_id` | string | Active task (`task_1` / `task_2` / `task_3`) |
| `status` | enum | `idle` \| `open` \| `resolved` \| `escalated` \| `timeout` |
| `sentiment` | enum | `positive` \| `neutral` \| `frustrated` \| `angry` |
| `priority` | enum | `low` \| `medium` \| `high` \| `urgent` |
| `category` | enum | `auth` \| `billing` \| `fulfillment` \| `bug` \| `sales` |
| `turn` | int | Current turn number |
| `max_turns` | int | Maximum turns before timeout |
| `history` | Message[] | Full conversation: `{role, text, turn}` |
| `kb_results` | string[] | KB articles retrieved (empty until `search_kb` called) |
| `kb_searched` | bool | Whether KB has been consulted |
| `empathized` | bool | Whether agent expressed empathy |
| `clarified` | bool | Whether agent asked a clarifying question |
| `solution_offered` | bool | Whether a solution has been offered |
| `escalated` | bool | Whether ticket was escalated |
| `cumulative_reward` | float | Running total reward |
| `done` | bool | Episode termination flag |

---

## Action Space

| Action | Payload | Reward | Notes |
|--------|---------|--------|-------|
| `search_kb` | — | **+2.0** | Retrieves KB articles for this ticket's category. Penalty −1.0 on duplicate. |
| `empathize` | — | **+1.0** | Acknowledges customer frustration. Zero reward on repeat. |
| `ask_clarify` | question text | **+1.0** | Requests more detail. Zero reward on repeat. |
| `offer_solution` | solution text | **+3.0 × quality** | Solution is scored against expected keywords. Penalty −1.0 if KB not searched first. |
| `escalate` | — | **−1.0** | Transfers to tier-2. Penalised to incentivise in-tier resolution. |
| `resolve` | — | **+5.0 + CSAT×2** | Ends episode. Penalty −3.0 if no solution offered. |
| `send_message` | message text | **+0.5** | Generic message. Useful for multi-turn clarification. |

### Reward decomposition
Every `Reward` object includes:
- `total` — net step reward
- `process_score` — correct action sequencing (0–1)
- `quality_score` — solution quality (0–1)
- `efficiency_score` — steps taken vs. optimal (0–1)
- `csat_score` — synthetic customer satisfaction (0–1)
- `penalties` — total penalties this step

---

## Tasks

### Task 1 — Easy: Resolve a Standard Auth Ticket
- **Ticket**: TKT-001 (account lockout, frustrated customer)
- **Max turns**: 8
- **Optimal policy**: `search_kb → empathize → offer_solution → resolve`
- **Max reward**: ~11.0
- **Grader weights**: KB searched (0.30), empathy (0.25), solution quality (0.25), resolved (0.20)

### Task 2 — Medium: Handle a Billing Dispute
- **Ticket**: TKT-003 (wrong invoice amount after plan downgrade)
- **Max turns**: 10
- **Optimal policy**: `search_kb → ask_clarify → empathize → offer_solution → resolve`
- **Challenge**: Generic solutions penalised; agent must cite a specific dollar credit.
- **Grader weights**: clarify (0.20), KB (0.20), solution quality (0.30), empathy (0.15), resolved (0.15)

### Task 3 — Hard: Triage a Critical Time-Sensitive Bug
- **Ticket**: TKT-006 (data export stuck, compliance deadline tomorrow)
- **Max turns**: 8
- **Optimal policy**: `search_kb → empathize → ask_clarify → offer_solution → resolve`
- **Challenge**: Two-part solution required (priority queue + partial export). Escalation is capped. Score requires urgency awareness.
- **Grader weights**: KB (0.20), empathy (0.15), two-part solution (0.35), no escalation (0.15), resolved (0.15)

---

## Reward Function Design

The reward function encodes three business objectives simultaneously:

1. **Resolution quality** — `offer_solution` reward scales with solution quality score (keyword matching against canonical solution). Forces the agent to consult the KB before improvising.

2. **Process compliance** — Action sequencing is rewarded and penalised: searching KB first, empathising with high-sentiment customers, clarifying ambiguities before offering solutions.

3. **Customer experience** — The CSAT bonus on `resolve` (up to +2.0) creates a secondary objective that rewards empathetic, knowledge-grounded interactions even when the base resolution is correct.

### Shaped vs. sparse
Reward is **dense** — every action produces a signal. The agent never needs to reach `resolve` to receive useful gradient. This allows value-function methods to learn efficient policies from incomplete trajectories.

---

## Grader Specification

All graders are **deterministic**: identical observations produce identical scores.

- Scores are in `[0.0, 1.0]`
- Each grader inspects the final `Observation`: flags (`kb_searched`, `empathized`, `clarified`, `solution_offered`, `escalated`, `status`) and conversation `history`
- Solution quality is measured by keyword presence in agent turn text
- **Pass threshold**: ≥ 0.70 on all tasks

---

## Baseline Scores

| Task | Difficulty | Model | Grader Score | Passed |
|------|-----------|-------|-------------|--------|
| task_1 | easy | gpt-4o-mini | 0.85 | ✓ |
| task_2 | medium | gpt-4o-mini | 0.78 | ✓ |
| task_3 | hard | gpt-4o-mini | 0.65 | — |
| **avg** | | | **0.76** | |

---

## Project Structure

```
customer_support_env/
├── server.py              # FastAPI app — /reset, /step, /state, /grade
├── inference.py           # Baseline inference script (OpenAI client)
├── openenv.yaml           # OpenEnv spec file
├── requirements.txt
├── Dockerfile
├── README.md
├── env/
│   ├── __init__.py
│   ├── models.py          # Typed Pydantic models: Observation, Action, Reward
│   ├── environment.py     # Core CustomerSupportEnv class
│   └── tickets.py         # Ticket scenario database (6 tickets, KB articles)
├── graders/
│   ├── __init__.py
│   └── graders.py         # Programmatic graders for all 3 tasks
└── tests/
    ├── __init__.py
    └── test_env.py        # 25 unit tests
```

---

## Running Tests

```bash
pytest tests/ -v
```

Or without pytest:
```bash
python -m tests.test_env
```

---

## Hugging Face Space Configuration

Add the following to the top of `README.md` for HF Spaces auto-detection:

```yaml
---
title: CustomerSupportEnv
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - nlp
---
```

---

## License

MIT
