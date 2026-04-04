"""
server.py — FastAPI/OpenEnv server wrapper for CustomerSupportEnv.

Exposes the environment as REST endpoints compatible with OpenEnv specification.
Handles session management and action validation.

Endpoints:
  POST   /reset              → Initialize new episode, return initial observation
  POST   /step               → Apply action, return (obs, reward, done)
  GET    /state              → Get current environment state
  GET    /tasks              → List all tasks
  POST   /grade              → Grade current episode
  GET    /health             → Health check
  GET    /openenv.yaml       → Spec file
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, Optional
from pathlib import Path

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Request, Body
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, ConfigDict
    import uvicorn
except ImportError as e:
    print(f"[ERROR] Missing FastAPI dependency: {e}", flush=True)
    print("Run: pip install fastapi uvicorn pydantic", flush=True)
    sys.exit(1)

# Local env imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from env.environment import CustomerSupportEnv, TASKS
    from env.models import Action, ActionType, Observation, Reward
    from graders.graders import grade
except ImportError as e:
    print(f"[ERROR] Missing local env module: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="CustomerSupportEnv",
    description="OpenEnv-compatible customer support RL environment",
    version="1.0.0"
)

# ── Session Storage (in-memory for single deployment) ───────────────────────
_sessions: Dict[str, Dict[str, Any]] = {}
_session_counter = 0


def new_session_id() -> str:
    """Generate a unique session ID."""
    global _session_counter
    _session_counter += 1
    return f"session_{_session_counter:06d}"


# ── Pydantic Models ──────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    session_id: str
    action_type: str
    payload: Optional[str] = None


class GradeRequest(BaseModel):
    session_id: str


# ── Helper: Make JSON serializable ──────────────────────────────────────────
def to_json_serializable(obj: Any) -> Any:
    """Convert any object to JSON-serializable format."""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif hasattr(obj, 'dict') and callable(obj.dict):
        # Pydantic model
        return to_json_serializable(obj.dict())
    elif hasattr(obj, '__dict__'):
        # Regular object with attributes
        return to_json_serializable(obj.__dict__)
    else:
        # Fallback to string representation
        return str(obj)


def serialize_obs(obs: Observation) -> Dict[str, Any]:
    """Convert Observation dataclass to JSON-serializable dict."""
    # Convert all fields to JSON-serializable format
    return {
        "ticket_id": to_json_serializable(obs.ticket_id),
        "task_id": to_json_serializable(obs.task_id),
        "status": to_json_serializable(obs.status),
        "sentiment": to_json_serializable(obs.sentiment),
        "priority": to_json_serializable(obs.priority),
        "category": to_json_serializable(obs.category),
        "turn": to_json_serializable(obs.turn),
        "max_turns": to_json_serializable(obs.max_turns),
        "history": to_json_serializable(obs.history),
        "kb_results": to_json_serializable(obs.kb_results),
        "kb_searched": to_json_serializable(obs.kb_searched),
        "empathized": to_json_serializable(obs.empathized),
        "clarified": to_json_serializable(obs.clarified),
        "solution_offered": to_json_serializable(obs.solution_offered),
        "escalated": to_json_serializable(obs.escalated),
        "cumulative_reward": to_json_serializable(obs.cumulative_reward),
        "done": to_json_serializable(obs.done),
    }


def serialize_reward(reward: Reward) -> Dict[str, Any]:
    """Convert Reward dataclass to JSON-serializable dict."""
    return {
        "total": to_json_serializable(reward.total),
        "breakdown": to_json_serializable(reward.breakdown),
        "reason": to_json_serializable(reward.reason),
    }


# ── OpenEnv Endpoints ────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(request: Optional[Dict[str, Any]] = Body(default=None)) -> JSONResponse:
    """
    Reset environment and start a new episode.
    
    Accepts both empty POST and JSON body with optional parameters.
    
    Args:
        task_id: One of task_1, task_2, task_3 (optional, defaults to task_1)
        seed: Optional random seed (defaults to 42)
    
    Returns:
        {
            "session_id": str,
            "observation": {...},
            "info": {...}
        }
    """
    try:
        # Default values
        task_id = "task_1"
        seed = 42
        
        # Override with request values if provided
        if request is not None and isinstance(request, dict):
            if "task_id" in request and request["task_id"]:
                task_id = request["task_id"]
            if "seed" in request and request["seed"] is not None:
                seed = request["seed"]
        
        print(f"[RESET] task_id={task_id}, seed={seed}", flush=True)
        
        # Validate task_id
        if task_id not in TASKS:
            raise ValueError(f"Invalid task_id '{task_id}'. Must be one of: {list(TASKS.keys())}")
        
        # Create and reset environment
        env = CustomerSupportEnv(task_id=task_id, seed=seed)
        obs = env.reset()
        
        # Store session
        session_id = new_session_id()
        _sessions[session_id] = {
            "env": env,
            "task_id": task_id,
            "observation": obs,
            "steps": 0,
            "done": False,
        }
        
        print(f"[RESET] Created session {session_id}", flush=True)
        
        # Serialize observation to ensure JSON compatibility
        obs_json = serialize_obs(obs)
        
        return JSONResponse(
            status_code=200,
            content={
                "session_id": session_id,
                "observation": obs_json,
                "info": {
                    "task_id": task_id,
                    "difficulty": TASKS[task_id].difficulty,
                    "description": TASKS[task_id].description,
                }
            }
        )
    
    except ValueError as e:
        print(f"[RESET ERROR] Validation error: {e}", flush=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[RESET ERROR] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(request: StepRequest) -> JSONResponse:
    """
    Apply an action and step the environment.
    
    Args:
        session_id: Session ID from /reset
        action_type: One of [search_kb, empathize, ask_clarify, offer_solution, escalate, resolve, send_message]
        payload: Optional action payload (required for some action types)
    
    Returns:
        {
            "observation": {...},
            "reward": {...},
            "done": bool,
            "info": {...}
        }
    """
    try:
        session_id = request.session_id
        action_type = request.action_type
        payload = request.payload
        
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        
        session = _sessions[session_id]
        env = session["env"]
        
        if session["done"]:
            raise HTTPException(status_code=400, detail="Episode already done. Call /reset to start new episode.")
        
        # Create action
        action = Action(action_type=action_type, payload=payload)
        
        # Step environment
        result = env.step(action)
        
        # Update session
        session["observation"] = result.observation
        session["steps"] += 1
        session["done"] = result.observation.done
        
        # Serialize for JSON compatibility
        obs_json = serialize_obs(result.observation)
        reward_json = serialize_reward(result.reward)
        
        return JSONResponse(
            status_code=200,
            content={
                "observation": obs_json,
                "reward": reward_json,
                "done": result.observation.done,
                "info": {
                    "step": session["steps"],
                    "action": action_type,
                }
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
async def state_endpoint(session_id: str) -> JSONResponse:
    """
    Get current environment state without stepping.
    
    Args:
        session_id: Session ID from /reset
    
    Returns:
        {
            "observation": {...},
            "info": {...}
        }
    """
    try:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        
        session = _sessions[session_id]
        obs = session["observation"]
        
        obs_json = serialize_obs(obs)
        
        return JSONResponse(
            status_code=200,
            content={
                "observation": obs_json,
                "info": {
                    "task_id": session["task_id"],
                    "steps": session["steps"],
                    "done": session["done"],
                }
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"State query failed: {str(e)}")


@app.get("/tasks")
async def tasks_endpoint() -> JSONResponse:
    """
    List all available tasks.
    
    Returns:
        {
            "tasks": [
                {
                    "id": "task_1",
                    "name": "...",
                    "difficulty": "easy|medium|hard",
                    "description": "...",
                    "max_turns": int
                },
                ...
            ]
        }
    """
    try:
        task_list = []
        for task_id, task_obj in TASKS.items():
            task_list.append({
                "id": task_id,
                "name": task_obj.name,
                "difficulty": task_obj.difficulty,
                "description": task_obj.description,
                "max_turns": task_obj.max_turns,
            })
        
        return JSONResponse(
            status_code=200,
            content={"tasks": task_list}
        )
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Tasks query failed: {str(e)}")


@app.post("/grade")
async def grade_endpoint(request: GradeRequest) -> JSONResponse:
    """
    Grade the current episode.
    
    Args:
        session_id: Session ID from /reset
    
    Returns:
        {
            "score": float (0.0 to 1.0),
            "passed": bool,
            "breakdown": {...},
            "reason": str
        }
    """
    try:
        session_id = request.session_id
        
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        
        session = _sessions[session_id]
        env = session["env"]
        task_id = session["task_id"]
        
        # Get final state
        final_obs = env.state()
        
        # Grade
        grader_result = grade(task_id, final_obs)
        
        return JSONResponse(
            status_code=200,
            content={
                "score": grader_result.score,
                "passed": grader_result.passed,
                "breakdown": to_json_serializable(grader_result.breakdown),
                "reason": grader_result.reason,
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "CustomerSupportEnv",
            "version": "1.0.0",
            "sessions_active": len(_sessions),
        }
    )


@app.get("/openenv.yaml")
async def openenv_spec() -> FileResponse:
    """Serve OpenEnv specification."""
    spec_path = Path(__file__).parent / "openenv.yaml"
    if not spec_path.exists():
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    return FileResponse(spec_path, media_type="text/yaml")


# ── Root endpoint ────────────────────────────────────────────────────────────
@app.get("/")
async def root() -> JSONResponse:
    """Root endpoint."""
    return JSONResponse(
        status_code=200,
        content={
            "service": "CustomerSupportEnv OpenEnv Server",
            "version": "1.0.0",
            "endpoints": {
                "POST /reset": "Initialize new episode",
                "POST /step": "Apply action",
                "GET /state": "Get current state",
                "GET /tasks": "List tasks",
                "POST /grade": "Grade episode",
                "GET /health": "Health check",
                "GET /openenv.yaml": "Specification",
            }
        }
    )


# ── Startup/Shutdown ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Log startup."""
    print("[INFO] CustomerSupportEnv server started", flush=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown."""
    print("[INFO] CustomerSupportEnv server shutdown", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    """Entry point for the server."""
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"[INFO] Starting server on {host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
