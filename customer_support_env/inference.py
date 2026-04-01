"""
inference.py — Baseline inference script for CustomerSupportEnv.

Runs an LLM agent against all 3 tasks using the OpenAI client.
Emits structured stdout logs in the required [START]/[STEP]/[END] format.

Environment variables required:
  API_BASE_URL   The API endpoint for the LLM (e.g. https://api.openai.com/v1)
  MODEL_NAME     The model identifier (e.g. gpt-4o-mini)
  HF_TOKEN       Your Hugging Face / API key

Usage:
  python inference.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ── OpenAI client (uses env vars) ─────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed. Run: pip install openai", flush=True)
    sys.exit(1)

# ── Local env imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.environment import CustomerSupportEnv, TASKS
from env.models import Action, ActionType
from graders.graders import grade

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN (or OPENAI_API_KEY) environment variable not set.", flush=True)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Action schema for structured output ──────────────────────────────────────
VALID_ACTIONS = ["search_kb", "empathize", "ask_clarify", "offer_solution", "escalate", "resolve", "send_message"]

SYSTEM_PROMPT = """You are a customer support AI agent operating inside a reinforcement learning environment.

On each turn you will receive:
- The current ticket details (category, priority, sentiment)
- The conversation history
- Any KB articles already retrieved
- Your cumulative reward so far

Your goal is to MAXIMISE the episode reward by following best practice:
1. Always call search_kb first to retrieve relevant knowledge base articles.
2. Empathise with frustrated or angry customers before diving into solutions.
3. Clarify details when information is ambiguous.
4. Offer a specific, concrete solution using information from the KB articles.
5. Resolve the ticket cleanly. Do NOT escalate unless truly unavoidable.

Respond ONLY with a valid JSON object (no markdown, no extra text):
{
  "action_type": "<one of: search_kb | empathize | ask_clarify | offer_solution | escalate | resolve | send_message>",
  "payload": "<optional: your message or solution text, required for offer_solution/send_message/ask_clarify>"
}"""


def build_user_message(obs_dict: Dict[str, Any]) -> str:
    history_text = ""
    for msg in obs_dict.get("history", []):
        role = msg.get("role", "")
        text = msg.get("text", "")
        history_text += f"  [{role.upper()}]: {text}\n"

    kb_text = ""
    for article in obs_dict.get("kb_results", []):
        kb_text += f"  - {article}\n"

    return f"""Current ticket state:
  Ticket ID : {obs_dict.get('ticket_id')}
  Category  : {obs_dict.get('category')}
  Priority  : {obs_dict.get('priority')}
  Sentiment : {obs_dict.get('sentiment')}
  Turn      : {obs_dict.get('turn')} / {obs_dict.get('max_turns')}
  Cumulative reward: {obs_dict.get('cumulative_reward')}

Conversation history:
{history_text or '  (no messages yet)'}

KB articles retrieved:
{kb_text or '  (none — call search_kb to retrieve)'}

KB searched: {obs_dict.get('kb_searched')}
Empathized : {obs_dict.get('empathized')}
Clarified  : {obs_dict.get('clarified')}
Solution offered: {obs_dict.get('solution_offered')}

What is your next action?"""


def call_llm(messages: List[Dict]) -> Dict[str, str]:
    """Call the LLM and parse the JSON action response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract JSON from response
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        parsed = json.loads(m.group()) if m else {"action_type": "search_kb", "payload": None}
    return parsed


def run_task(task_id: str) -> Dict[str, Any]:
    """Run the agent on one task and return results."""
    env = CustomerSupportEnv(task_id=task_id, seed=42)
    obs = env.reset()
    obs_dict = obs.dict()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(json.dumps({
        "event": "START",
        "task_id": task_id,
        "ticket_id": obs_dict["ticket_id"],
        "difficulty": TASKS[task_id].difficulty,
        "model": MODEL_NAME,
    }), flush=True)

    episode_rewards = []
    step_num = 0

    while not obs_dict.get("done", False):
        step_num += 1
        user_msg = build_user_message(obs_dict)
        messages.append({"role": "user", "content": user_msg})

        # LLM inference
        try:
            action_dict = call_llm(messages)
        except Exception as e:
            print(f"[LLM ERROR] {e}", flush=True)
            action_dict = {"action_type": "resolve", "payload": None}

        action_type = action_dict.get("action_type", "resolve")
        payload = action_dict.get("payload")

        # Validate action
        if action_type not in VALID_ACTIONS:
            action_type = "search_kb"

        action = Action(action_type=action_type, payload=payload)

        try:
            result = env.step(action)
        except RuntimeError as e:
            print(f"[ENV ERROR] {e}", flush=True)
            break

        obs_dict = result.observation.dict()
        reward_dict = result.reward.dict()
        episode_rewards.append(reward_dict["total"])

        # Append assistant response to message history
        messages.append({
            "role": "assistant",
            "content": json.dumps(action_dict)
        })

        print(json.dumps({
            "event": "STEP",
            "task_id": task_id,
            "step": step_num,
            "action_type": action_type,
            "reward": reward_dict["total"],
            "cumulative_reward": obs_dict["cumulative_reward"],
            "done": obs_dict["done"],
            "reason": reward_dict.get("reason", ""),
        }), flush=True)

        if obs_dict.get("done"):
            break

    # Grade the episode
    final_obs = env.state()
    grader_result = grade(task_id, final_obs)

    print(json.dumps({
        "event": "END",
        "task_id": task_id,
        "difficulty": TASKS[task_id].difficulty,
        "total_steps": step_num,
        "cumulative_reward": obs_dict.get("cumulative_reward", 0),
        "grader_score": grader_result.score,
        "grader_passed": grader_result.passed,
        "grader_breakdown": grader_result.breakdown,
        "grader_reason": grader_result.reason,
        "final_status": obs_dict.get("status"),
    }), flush=True)

    return {
        "task_id": task_id,
        "difficulty": TASKS[task_id].difficulty,
        "grader_score": grader_result.score,
        "passed": grader_result.passed,
        "steps": step_num,
        "cumulative_reward": obs_dict.get("cumulative_reward", 0),
    }


def main():
    all_results = []

    for task_id in ["task_1", "task_2", "task_3"]:
        result = run_task(task_id)
        all_results.append(result)
        time.sleep(1)  # Avoid rate limiting

    # Summary
    avg_score = sum(r["grader_score"] for r in all_results) / len(all_results)
    print(json.dumps({
        "event": "SUMMARY",
        "model": MODEL_NAME,
        "results": all_results,
        "average_grader_score": round(avg_score, 3),
        "tasks_passed": sum(1 for r in all_results if r["passed"]),
        "total_tasks": len(all_results),
    }), flush=True)


if __name__ == "__main__":
    main()
