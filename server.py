"""
CustomerSupportEnv — FastAPI server.

Endpoints:
  POST /reset              → Observation
  POST /step               → StepResult
  GET  /state              → Observation
  GET  /tasks              → list of task specs
  POST /grade              → GraderResult
  GET  /health             → 200 OK
  GET  /openenv.yaml       → spec file
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from env.environment import CustomerSupportEnv, TASKS
from env.models import Action, Observation, StepResult, GraderResult
from graders.graders import grade

app = FastAPI(
    title="CustomerSupportEnv",
    description="OpenEnv-compatible RL environment for customer support agent training.",
    version="1.0.0",
)

# One env instance per task (keyed by task_id)
_envs: dict[str, CustomerSupportEnv] = {}


def _get_env(task_id: str) -> CustomerSupportEnv:
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    if task_id not in _envs:
        _envs[task_id] = CustomerSupportEnv(task_id=task_id)
    return _envs[task_id]


class ResetRequest(BaseModel):
    task_id: str = "task_1"


class StepRequest(BaseModel):
    task_id: str = "task_1"
    action_type: str
    payload: Optional[str] = None


class GradeRequest(BaseModel):
    task_id: str


@app.get("/health")
def health():
    return {"status": "ok", "version": CustomerSupportEnv.VERSION}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    env = _get_env(req.task_id)
    obs = env.reset()
    return obs


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    env = _get_env(req.task_id)
    try:
        action = Action(action_type=req.action_type, payload=req.payload)
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state", response_model=Observation)
def state(task_id: str = "task_1"):
    env = _get_env(task_id)
    return env.state()


@app.get("/tasks")
def list_tasks():
    return {tid: spec.dict() for tid, spec in TASKS.items()}


@app.post("/grade", response_model=GraderResult)
def grade_endpoint(req: GradeRequest):
    env = _get_env(req.task_id)
    obs = env.state()
    result = grade(req.task_id, obs)
    return result


@app.get("/openenv.yaml")
def get_yaml():
    yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    if os.path.exists(yaml_path):
        return FileResponse(yaml_path, media_type="text/yaml")
    return JSONResponse({"error": "openenv.yaml not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Customer Support Env 🚀</h1>
    <p>API is running successfully.</p>
    <ul>
        <li>/reset</li>
        <li>/step</li>
        <li>/state</li>
        <li>/grade</li>
    </ul>
    """