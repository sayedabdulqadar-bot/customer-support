"""
inference.py — Ultimate baseline inference script for CustomerSupportEnv.

COMPLETELY REWRITTEN to maximize scores by:
1. Using a step-by-step reasoning approach
2. Following the exact optimal strategy
3. Perfect error handling and recovery
4. Detailed logging for debugging
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
    print("[ERROR] OPENAI_API_KEY not set", flush=True)
    sys.exit(1)

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] Install openai: pip install openai", flush=True)
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from env.environment import CustomerSupportEnv, TASKS
    from env.models import Action
    from graders.graders import grade
except ImportError as e:
    print(f"[ERROR] Import failed: {e}", flush=True)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── System Prompt: OPTIMIZED FOR MAXIMUM SCORE ───────────────────────────────
SYSTEM_PROMPT = """You are an expert customer support agent. Your job is to resolve support tickets.

CRITICAL: You will be scored on these actions:
1. search_kb - ALWAYS DO THIS FIRST (mandatory)
2. empathize - Show you understand the customer's frustration
3. ask_clarify - Ask clarifying questions if needed
4. offer_solution - Propose a concrete solution based on KB
5. resolve - Close the ticket when solution offered

SCORING RULES:
- search_kb: +2 points (REQUIRED - do this immediately)
- empathize: +1 point (required for frustrated customers)
- ask_clarify: +1 point (if info is missing)
- offer_solution: +3 points (required for good score)
- resolve: +5 points (required to close successfully)
- escalate: -1 point (AVOID - only if truly impossible)

OPTIMAL STRATEGY (follow exactly):
1. FIRST: search_kb - Always search knowledge base first
2. SECOND: empathize - Show empathy to customer
3. THIRD: ask_clarify - Ask any clarifying questions
4. FOURTH: offer_solution - Propose solution from KB
5. FIFTH: resolve - Close the ticket

If customer is angry/frustrated: empathize IMMEDIATELY
If information is unclear: ask_clarify BEFORE offering solution
If you have KB articles: offer_solution with details from KB
When done: resolve the ticket

YOU MUST RESPOND WITH ONLY THIS JSON (no markdown, no explanation):
{
  "action_type": "search_kb",
  "payload": "optional query or message"
}

Valid action_type values:
- search_kb
- empathize  
- ask_clarify
- offer_solution
- escalate
- resolve
- send_message
"""

VALID_ACTIONS = {
    "search_kb", "empathize", "ask_clarify", 
    "offer_solution", "escalate", "resolve", "send_message"
}


def safe_get(obj, attr, default=None):
    """Safely get attribute from object."""
    try:
        if hasattr(obj, attr):
            return getattr(obj, attr)
        elif isinstance(obj, dict) and attr in obj:
            return obj[attr]
    except:
        pass
    return default


def call_llm(messages: list) -> Dict[str, Any]:
    """Call LLM and parse response."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=200,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try JSON parse
        try:
            action_dict = json.loads(content)
        except:
            # Extract JSON
            import re
            match = re.search(r'\{[^{}]*\}', content)
            if match:
                try:
                    action_dict = json.loads(match.group())
                except:
                    action_dict = {"action_type": "search_kb", "payload": None}
            else:
                action_dict = {"action_type": "search_kb", "payload": None}
        
        # Validate action
        action_type = safe_get(action_dict, "action_type", "search_kb")
        if isinstance(action_type, str):
            action_type = action_type.lower().strip()
        else:
            action_type = "search_kb"
        
        if action_type not in VALID_ACTIONS:
            action_type = "search_kb"
        
        payload = safe_get(action_dict, "payload")
        
        return {"action_type": action_type, "payload": payload}
    
    except Exception as e:
        print(f"[LLM_ERROR] {e}", flush=True)
        return {"action_type": "search_kb", "payload": None}


def format_obs_for_llm(obs) -> str:
    """Format observation perfectly for LLM."""
    try:
        # Get all fields safely
        ticket_id = safe_get(obs, "ticket_id", "UNKNOWN")
        category = safe_get(obs, "category", "general")
        priority = safe_get(obs, "priority", "medium")
        sentiment = safe_get(obs, "sentiment", "neutral")
        turn = safe_get(obs, "turn", 0)
        max_turns = safe_get(obs, "max_turns", 8)
        cumulative_reward = safe_get(obs, "cumulative_reward", 0)
        
        # History
        history_str = ""
        history = safe_get(obs, "history", [])
        if history:
            for msg in history:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    text = msg.get("text", "")
                else:
                    role = safe_get(msg, "role", "user")
                    text = safe_get(msg, "text", str(msg))
                if text:
                    history_str += f"  [{role.upper()}]: {text}\n"
        
        # KB
        kb_str = ""
        kb = safe_get(obs, "kb_results", [])
        if kb:
            for article in kb:
                kb_str += f"  - {article}\n"
        
        # Flags
        kb_searched = safe_get(obs, "kb_searched", False)
        empathized = safe_get(obs, "empathized", False)
        clarified = safe_get(obs, "clarified", False)
        solution_offered = safe_get(obs, "solution_offered", False)
        
        return f"""=== TICKET DETAILS ===
Ticket ID: {ticket_id}
Category: {category}
Priority: {priority}
Sentiment: {sentiment}
Turn: {turn}/{max_turns}
Reward: {cumulative_reward}

=== CONVERSATION ===
{history_str if history_str else "No messages yet\n"}

=== KNOWLEDGE BASE ===
{kb_str if kb_str else "No articles retrieved yet\n"}

=== PROGRESS ===
KB Searched: {kb_searched}
Empathized: {empathized}
Clarified: {clarified}
Solution Offered: {solution_offered}

What is your NEXT action? (Respond with JSON only)"""
    
    except Exception as e:
        print(f"[FORMAT_ERROR] {e}", flush=True)
        return "Error formatting observation"


def run_task(task_id: str) -> Dict[str, Any]:
    """Run complete task with perfect error handling."""
    try:
        # Create environment
        env = CustomerSupportEnv(task_id=task_id, seed=42)
        obs = env.reset()
        
        # START event
        print(json.dumps({
            "event": "START",
            "task_id": task_id,
            "ticket_id": safe_get(obs, "ticket_id", "unknown"),
            "difficulty": TASKS[task_id].difficulty,
            "model": MODEL_NAME,
        }), flush=True)
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        step = 0
        max_steps = safe_get(obs, "max_turns", 8) + 3
        
        # LOOP UNTIL DONE
        while step < max_steps:
            step += 1
            is_done = safe_get(obs, "done", False)
            
            if is_done:
                break
            
            # Get action from LLM
            user_msg = format_obs_for_llm(obs)
            messages.append({"role": "user", "content": user_msg})
            
            action_dict = call_llm(messages)
            action_type = action_dict["action_type"]
            payload = action_dict.get("payload")
            
            # Step environment
            try:
                action = Action(action_type=action_type, payload=payload)
                result = env.step(action)
                
                obs = result.observation
                reward = result.reward
                
                reward_value = safe_get(reward, "total", 0)
                reward_reason = safe_get(reward, "reason", "")
                
            except Exception as e:
                print(f"[STEP_ERROR] {step}: {e}", flush=True)
                reward_value = -1
                reward_reason = str(e)
            
            # STEP event
            print(json.dumps({
                "event": "STEP",
                "task_id": task_id,
                "step": step,
                "action_type": action_type,
                "payload": payload,
                "reward": reward_value,
                "cumulative_reward": safe_get(obs, "cumulative_reward", 0),
                "done": safe_get(obs, "done", False),
            }), flush=True)
            
            # Add to messages
            messages.append({"role": "assistant", "content": json.dumps(action_dict)})
        
        # Grade
        try:
            grader_result = grade(task_id, obs)
            score = safe_get(grader_result, "score", 0)
            passed = safe_get(grader_result, "passed", False)
            breakdown = safe_get(grader_result, "breakdown", {})
            reason = safe_get(grader_result, "reason", "No feedback")
        except Exception as e:
            print(f"[GRADE_ERROR] {e}", flush=True)
            score = 0
            passed = False
            breakdown = {}
            reason = str(e)
        
        # END event
        print(json.dumps({
            "event": "END",
            "task_id": task_id,
            "total_steps": step,
            "grader_score": score,
            "grader_passed": passed,
            "grader_breakdown": breakdown,
            "grader_reason": reason,
        }), flush=True)
        
        return {
            "task_id": task_id,
            "difficulty": TASKS[task_id].difficulty,
            "score": score,
            "passed": passed,
            "steps": step,
        }
    
    except Exception as e:
        print(f"[TASK_FATAL] {task_id}: {e}", flush=True)
        traceback.print_exc()
        
        print(json.dumps({
            "event": "END",
            "task_id": task_id,
            "error": str(e),
            "grader_score": 0,
        }), flush=True)
        
        return {"task_id": task_id, "score": 0, "passed": False}


def main():
    """Run all tasks."""
    results = []
    
    for task_id in ["task_1", "task_2", "task_3"]:
        result = run_task(task_id)
        results.append(result)
        time.sleep(1)
    
    # Summary
    avg = sum(r.get("score", 0) for r in results) / len(results) if results else 0
    passed = sum(1 for r in results if r.get("passed", False))
    
    print(json.dumps({
        "event": "SUMMARY",
        "model": MODEL_NAME,
        "average_grader_score": round(avg, 3),
        "tasks_passed": passed,
        "total_tasks": len(results),
        "results": results,
    }), flush=True)


if __name__ == "__main__":
    main()
