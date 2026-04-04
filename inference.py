"""
inference.py — Fixed baseline inference script for CustomerSupportEnv.

Runs an LLM agent against all 3 tasks with proper error handling and logging.
Outputs structured stdout logs in [START]/[STEP]/[END] format.

Required environment variables:
  API_BASE_URL   The LLM API endpoint (default: https://api.openai.com/v1)
  MODEL_NAME     The LLM model identifier (default: gpt-4o-mini)
  OPENAI_API_KEY Your OpenAI API key (required)
"""

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional

# ── Environment Variables ────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")

if not API_KEY:
    print("[ERROR] OPENAI_API_KEY environment variable not set", flush=True)
    sys.exit(1)

# ── OpenAI Client ────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed. Run: pip install openai", flush=True)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Local Imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from env.environment import CustomerSupportEnv, TASKS
    from env.models import Action
    from graders.graders import grade
except ImportError as e:
    print(f"[ERROR] Failed to import local modules: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a customer support AI agent operating inside a reinforcement learning environment.

You will receive observations containing:
- Current ticket details (category, priority, sentiment)
- Conversation history
- Knowledge base articles retrieved
- Your cumulative reward so far

Your goal is to MAXIMIZE the episode reward by following best practices:

1. **ALWAYS call search_kb first** to retrieve knowledge base articles
2. **Empathize** with frustrated or angry customers before solutions
3. **Clarify** ambiguous details before offering solutions
4. **Offer concrete solutions** using KB articles
5. **Resolve** the ticket when you have enough information
6. **Avoid escalation** unless absolutely necessary

You must respond with ONLY a valid JSON object (no markdown, no extra text):

{
  "action_type": "search_kb|empathize|ask_clarify|offer_solution|escalate|resolve|send_message",
  "payload": "optional message or solution text (required for some actions)"
}

IMPORTANT:
- action_type MUST be one of the exact options listed
- Keep payload concise and relevant
- Each action should make progress toward resolution
- Think about what information you still need before offering solutions
"""

# ── Valid Actions ────────────────────────────────────────────────────────────
VALID_ACTIONS = {
    "search_kb",
    "empathize",
    "ask_clarify",
    "offer_solution",
    "escalate",
    "resolve",
    "send_message"
}

# ── Helper: Call LLM ─────────────────────────────────────────────────────────
def call_llm(messages: list) -> Dict[str, Any]:
    """Call the LLM and return parsed action."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=256,
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            action_dict = json.loads(raw_response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                action_dict = json.loads(json_match.group())
            else:
                # Fallback: search_kb is a safe default
                action_dict = {"action_type": "search_kb", "payload": None}
        
        # Validate action_type
        action_type = action_dict.get("action_type", "search_kb").lower().strip()
        if action_type not in VALID_ACTIONS:
            action_type = "search_kb"
        
        return {
            "action_type": action_type,
            "payload": action_dict.get("payload")
        }
    
    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)
        return {"action_type": "search_kb", "payload": None}


# ── Helper: Format Observation for LLM ───────────────────────────────────────
def format_observation(obs) -> str:
    """Format observation for LLM."""
    try:
        # Build history text
        history_text = ""
        if hasattr(obs, 'history') and obs.history:
            for msg in obs.history:
                if hasattr(msg, '__dict__'):
                    msg_dict = msg.__dict__
                elif isinstance(msg, dict):
                    msg_dict = msg
                else:
                    msg_dict = {"text": str(msg)}
                
                role = msg_dict.get("role", "user").upper()
                text = msg_dict.get("text", str(msg))
                history_text += f"  [{role}]: {text}\n"
        
        # Build KB results text
        kb_text = ""
        if hasattr(obs, 'kb_results') and obs.kb_results:
            for article in obs.kb_results:
                kb_text += f"  - {article}\n"
        
        # Build full message
        message = f"""Current Ticket State:
  ID: {obs.ticket_id}
  Category: {obs.category}
  Priority: {obs.priority}
  Sentiment: {obs.sentiment}
  Turn: {obs.turn}/{obs.max_turns}
  Reward so far: {obs.cumulative_reward}

Conversation History:
{history_text if history_text else "  (none yet)"}

Knowledge Base Articles Retrieved:
{kb_text if kb_text else "  (none - try search_kb)"}

Actions taken so far:
  - KB searched: {obs.kb_searched}
  - Empathized: {obs.empathized}
  - Clarified: {obs.clarified}
  - Solution offered: {obs.solution_offered}

What is your next action?"""
        
        return message
    
    except Exception as e:
        print(f"[FORMAT ERROR] {e}", flush=True)
        return "Error formatting observation"


# ── Main: Run Task ───────────────────────────────────────────────────────────
def run_task(task_id: str) -> Dict[str, Any]:
    """Run agent on a single task."""
    try:
        # Initialize environment
        env = CustomerSupportEnv(task_id=task_id, seed=42)
        obs = env.reset()
        
        print(json.dumps({
            "event": "START",
            "task_id": task_id,
            "ticket_id": obs.ticket_id,
            "difficulty": TASKS[task_id].difficulty,
            "model": MODEL_NAME,
        }), flush=True)
        
        # Initialize message history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        step_num = 0
        max_steps = obs.max_turns + 5  # Allow some buffer
        
        # Agent loop
        while step_num < max_steps and not obs.done:
            step_num += 1
            
            # Format observation for LLM
            user_message = format_observation(obs)
            messages.append({"role": "user", "content": user_message})
            
            # Get action from LLM
            action_dict = call_llm(messages)
            action_type = action_dict["action_type"]
            payload = action_dict.get("payload")
            
            # Create action
            action = Action(action_type=action_type, payload=payload)
            
            # Step environment
            try:
                result = env.step(action)
                obs = result.observation
                reward = result.reward
                
                reward_total = reward.total if hasattr(reward, 'total') else 0
                reward_reason = reward.reason if hasattr(reward, 'reason') else ""
                
            except Exception as e:
                print(f"[ENV ERROR] Step {step_num}: {e}", flush=True)
                reward_total = 0
                reward_reason = f"Error: {str(e)}"
                # Continue to next step instead of breaking
                continue
            
            # Log step
            print(json.dumps({
                "event": "STEP",
                "task_id": task_id,
                "step": step_num,
                "action_type": action_type,
                "reward": reward_total,
                "cumulative_reward": obs.cumulative_reward,
                "done": obs.done,
                "reason": reward_reason,
            }), flush=True)
            
            # Add assistant message to history
            messages.append({"role": "assistant", "content": json.dumps(action_dict)})
        
        # Grade the episode
        try:
            grader_result = grade(task_id, obs)
            grader_score = grader_result.score
            grader_passed = grader_result.passed
            grader_breakdown = grader_result.breakdown
            grader_reason = grader_result.reason
        except Exception as e:
            print(f"[GRADER ERROR] {e}", flush=True)
            grader_score = 0.0
            grader_passed = False
            grader_breakdown = {}
            grader_reason = f"Grading failed: {str(e)}"
        
        # Log end
        print(json.dumps({
            "event": "END",
            "task_id": task_id,
            "difficulty": TASKS[task_id].difficulty,
            "total_steps": step_num,
            "cumulative_reward": obs.cumulative_reward,
            "grader_score": grader_score,
            "grader_passed": grader_passed,
            "grader_breakdown": grader_breakdown,
            "grader_reason": grader_reason,
            "final_status": obs.status if hasattr(obs, 'status') else "unknown",
        }), flush=True)
        
        return {
            "task_id": task_id,
            "difficulty": TASKS[task_id].difficulty,
            "grader_score": grader_score,
            "passed": grader_passed,
            "steps": step_num,
            "cumulative_reward": obs.cumulative_reward,
        }
    
    except Exception as e:
        print(f"[TASK ERROR] {task_id}: {e}", flush=True)
        traceback.print_exc()
        
        print(json.dumps({
            "event": "END",
            "task_id": task_id,
            "error": str(e),
            "grader_score": 0.0,
            "grader_passed": False,
        }), flush=True)
        
        return {
            "task_id": task_id,
            "grader_score": 0.0,
            "passed": False,
            "steps": 0,
            "error": str(e),
        }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    """Run baseline on all tasks."""
    all_results = []
    
    for task_id in ["task_1", "task_2", "task_3"]:
        try:
            result = run_task(task_id)
            all_results.append(result)
        except Exception as e:
            print(f"[FATAL ERROR] {task_id}: {e}", flush=True)
            traceback.print_exc()
        
        time.sleep(1)  # Avoid rate limiting
    
    # Summary
    if all_results:
        avg_score = sum(r["grader_score"] for r in all_results) / len(all_results)
        tasks_passed = sum(1 for r in all_results if r.get("passed", False))
    else:
        avg_score = 0.0
        tasks_passed = 0
    
    print(json.dumps({
        "event": "SUMMARY",
        "model": MODEL_NAME,
        "results": all_results,
        "average_grader_score": round(avg_score, 3),
        "tasks_passed": tasks_passed,
        "total_tasks": len(all_results),
    }), flush=True)


if __name__ == "__main__":
    main()
