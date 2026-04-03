"""
FastAPI server for the Customer Support Ticket Resolution Environment.
Entry point for the Docker container.

Endpoints:
  GET  /           health check
  GET  /info       openenv.yaml metadata
  GET  /health     runtime status
  POST /reset      start new episode
  POST /step       execute one action
  GET  /state      current observation (no side effects)
  WS   /ws         WebSocket interface
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from customer_support_env import (
    CustomerSupportEnv,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResult,
    WSMessage,
    WSResponse,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Support Ticket Resolution Environment",
    description=(
        "An OpenEnv-compatible reinforcement learning environment where an AI agent "
        "resolves customer support tickets by classifying, prioritizing, responding, "
        "escalating, and closing them."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

env = CustomerSupportEnv()
yaml_meta: dict = {}


@app.on_event("startup")
async def startup() -> None:
    global yaml_meta
    yaml_path = Path("openenv.yaml")
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_meta = yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict:
    return {
        "name": "customer-support-ticket-resolution",
        "version": "1.0.0",
        "status": "ok",
        "description": (
            "OpenEnv environment for customer support ticket resolution. "
            "Tasks: easy, medium, hard."
        ),
    }


@app.get("/info")
async def info() -> dict:
    return yaml_meta if yaml_meta else {"error": "openenv.yaml not found"}


@app.get("/health")
async def health() -> dict:
    task_id: Optional[str] = None
    step: int = 0
    done: bool = False

    if env._task_instance is not None:
        task_id = env._task_instance.spec.task_id
        step = env._current_step
        done = env._done

    return {
        "status": "ok",
        "step": step,
        "task_id": task_id,
        "done": done,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = ResetRequest()) -> ResetResponse:
    obs = env.reset(task_id=request.task_id, seed=request.seed)
    return ResetResponse(
        observation=obs,
        task_id=env._task_instance.spec.task_id,  # type: ignore[union-attr]
        max_steps=env._task_instance.spec.max_steps,  # type: ignore[union-attr]
    )


@app.post("/step", response_model=StepResult)
async def step(request: StepRequest) -> StepResult:
    if env._task_instance is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )
    return env.step(request.action)


@app.get("/state")
async def state():
    if env._task_instance is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )
    return env.state()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    async def send_response(response: WSResponse) -> None:
        await websocket.send_text(response.model_dump_json())

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = WSMessage.model_validate_json(raw)
            except (ValidationError, json.JSONDecodeError) as exc:
                await send_response(
                    WSResponse(type="error", payload={"message": f"Invalid message format: {exc}"})
                )
                continue

            # ---- reset ----
            if msg.type == "reset":
                try:
                    payload = msg.payload or {}
                    task_id = payload.get("task_id")
                    seed = payload.get("seed")
                    obs = env.reset(task_id=task_id, seed=seed)
                    await send_response(
                        WSResponse(
                            type="reset_ack",
                            payload={
                                "observation": obs.model_dump(),
                                "task_id": env._task_instance.spec.task_id,  # type: ignore
                                "max_steps": env._task_instance.spec.max_steps,  # type: ignore
                            },
                        )
                    )
                except Exception as exc:
                    await send_response(
                        WSResponse(type="error", payload={"message": str(exc)})
                    )

            # ---- step ----
            elif msg.type == "step":
                if env._task_instance is None:
                    await send_response(
                        WSResponse(
                            type="error",
                            payload={"message": "Not initialized. Send reset first."},
                        )
                    )
                    continue
                try:
                    from customer_support_env import Action
                    action_data = (msg.payload or {}).get("action", {})
                    action = Action.model_validate(action_data)
                    result = env.step(action)
                    await send_response(
                        WSResponse(type="step_result", payload=result.model_dump())
                    )
                except (ValidationError, Exception) as exc:
                    await send_response(
                        WSResponse(type="error", payload={"message": str(exc)})
                    )

            # ---- state ----
            elif msg.type == "state":
                if env._task_instance is None:
                    await send_response(
                        WSResponse(
                            type="error",
                            payload={"message": "Not initialized. Send reset first."},
                        )
                    )
                    continue
                try:
                    obs = env.state()
                    await send_response(
                        WSResponse(type="state_result", payload={"observation": obs.model_dump()})
                    )
                except Exception as exc:
                    await send_response(
                        WSResponse(type="error", payload={"message": str(exc)})
                    )

            # ---- ping ----
            elif msg.type == "ping":
                await send_response(
                    WSResponse(
                        type="pong",
                        payload={"timestamp": datetime.now(timezone.utc).isoformat()},
                    )
                )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await send_response(
                WSResponse(type="error", payload={"message": f"Unexpected error: {exc}"})
            )
        except Exception:
            pass
