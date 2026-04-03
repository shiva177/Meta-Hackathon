"""
Inference Script — Customer Support Ticket Resolution Environment
=================================================================
Mandatory:
  - API_BASE_URL  : LLM API endpoint  (e.g. https://router.huggingface.co/v1)
  - MODEL_NAME    : Model identifier
  - HF_TOKEN      : Hugging Face / API key

Optional:
  - ENV_HTTP_URL  : Base URL of the running environment server (default: http://localhost:8000)
  - SEED          : Integer seed for reproducibility (default: 42)

Usage:
  python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "EMPTY")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "")
ENV_HTTP_URL: str = os.environ.get("ENV_HTTP_URL", "http://localhost:8000")
SEED: int = int(os.environ.get("SEED", "42"))

TEMPERATURE: float = 0.0     # deterministic sampling for reproducibility
MAX_TOKENS: int = 512
MAX_STEPS: int = 10          # per task; env enforces its own hard limit
TASKS: List[str] = ["easy", "medium", "hard"]

if not MODEL_NAME:
    print(
        "ERROR: MODEL_NAME environment variable is not set.\n"
        "Export MODEL_NAME=<model-id> before running.",
        file=sys.stderr,
    )
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Environment HTTP client helpers
# ---------------------------------------------------------------------------

def _post(path: str, payload: Dict) -> Dict:
    resp = requests.post(f"{ENV_HTTP_URL}{path}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _get(path: str) -> Dict:
    resp = requests.get(f"{ENV_HTTP_URL}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_reset(task_id: str, seed: int) -> Dict:
    return _post("/reset", {"task_id": task_id, "seed": seed})


def env_step(action: Dict) -> Dict:
    return _post("/step", {"action": action})


def env_state() -> Dict:
    return _get("/state")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert customer support agent operating in a structured environment.
    You must respond with a single JSON object representing your next action.

    The JSON must have an "action_type" field. Other fields depend on the action type:

    ACTION TYPES:
    - classify: {"action_type": "classify", "category": "<billing|technical|account|feature_request|abuse|general>"}
    - set_priority: {"action_type": "set_priority", "priority": "<critical|high|medium|low>"}
    - respond: {"action_type": "respond", "response_text": "<your message to customer>"}
    - escalate: {"action_type": "escalate", "escalation_tier": "<tier2|tier3|legal|executive>", "escalation_reason": "<reason>"}
    - lookup_policy: {"action_type": "lookup_policy", "policy_query": "<search keywords>"}
    - resolve: {"action_type": "resolve", "resolved_ticket_id": "<ticket_id>", "resolution_note": "<internal note>"}
    - bulk_triage: {"action_type": "bulk_triage", "triage_list": [{"ticket_id": "...", "category": "...", "priority": "..."}, ...]}

    IMPORTANT:
    - Return ONLY valid JSON, no explanations, no markdown code blocks.
    - For bulk_triage, include ALL tickets in triage_list.
    - For respond, write a professional, empathetic customer-facing message.
    - For resolve, write a detailed internal resolution note.
    - For lookup_policy, use relevant keywords from the ticket to find applicable policies.
""").strip()


def _format_ticket(ticket: Dict) -> str:
    return (
        f"  Ticket ID: {ticket.get('ticket_id', 'N/A')}\n"
        f"  Subject: {ticket.get('subject', 'N/A')}\n"
        f"  Body: {ticket.get('body', 'N/A')}\n"
        f"  Customer Tier: {ticket.get('customer_tier', 'N/A')}\n"
        f"  Channel: {ticket.get('channel', 'N/A')}\n"
        f"  Previous Contacts: {ticket.get('previous_contacts', 0)}\n"
        f"  Account Age (days): {ticket.get('account_age_days', 0)}"
    )


def build_user_prompt(obs: Dict, history: List[str]) -> str:
    tickets = obs.get("tickets", [])
    task_id = obs.get("task_id", "unknown")
    step = obs.get("step", 0)
    instructions = obs.get("instructions", "")
    action_feedback = obs.get("action_feedback")
    action_valid = obs.get("action_valid")
    policy_result = obs.get("policy_result")
    escalation_queue = obs.get("escalation_queue", [])
    resolved_tickets = obs.get("resolved_tickets", [])

    lines = [
        f"=== STEP {step} | TASK: {task_id.upper()} ===",
        "",
        instructions,
        "",
        "--- TICKETS ---",
    ]

    for i, ticket in enumerate(tickets, 1):
        lines.append(f"[Ticket {i}]")
        lines.append(_format_ticket(ticket))
        lines.append("")

    if action_feedback:
        valid_str = "✓" if action_valid else "✗"
        lines.append(f"--- LAST ACTION RESULT [{valid_str}] ---")
        lines.append(action_feedback)
        lines.append("")

    if policy_result and policy_result.get("matched_policy_id"):
        lines.append("--- POLICY LOOKUP RESULT ---")
        lines.append(f"  Policy: [{policy_result['matched_policy_id']}] {policy_result.get('policy_title', '')}")
        lines.append(f"  Confidence: {policy_result.get('confidence', 0):.2f}")
        body = policy_result.get("policy_body", "")
        if body:
            lines.append(f"  Content: {body[:300]}{'...' if len(body) > 300 else ''}")
        lines.append("")

    if escalation_queue:
        lines.append(f"Escalated tickets: {escalation_queue}")
    if resolved_tickets:
        lines.append(f"Resolved tickets: {resolved_tickets}")

    if history:
        lines.append("")
        lines.append("--- HISTORY (last 5 steps) ---")
        lines.extend(history[-5:])

    lines.append("")
    lines.append("Your next action (JSON only):")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call + action parsing
# ---------------------------------------------------------------------------

def call_llm(user_prompt: str) -> str:
    """Call the LLM and return the raw response text."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [LLM ERROR] {exc}")
        return ""


def parse_action(text: str) -> Optional[Dict]:
    """
    Extract a JSON action from the LLM response.
    Tries: direct parse, then ```json ... ``` block, then first {...} found.
    """
    text = text.strip()
    if not text:
        return None

    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Try ```json ... ``` block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict) and "action_type" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # Try first {...} in response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "action_type" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    return None


def fallback_action(obs: Dict) -> Dict:
    """Return a safe fallback action based on current observation."""
    task_id = obs.get("task_id", "easy")
    actions_in_history = obs.get("resolved_tickets", [])

    if task_id == "medium":
        tickets = obs.get("tickets", [])
        return {
            "action_type": "bulk_triage",
            "triage_list": [
                {
                    "ticket_id": t["ticket_id"],
                    "category": "general",
                    "priority": "medium",
                }
                for t in tickets
            ],
        }
    if task_id == "hard":
        tickets = obs.get("tickets", [])
        tid = tickets[0]["ticket_id"] if tickets else "TKT-001"
        return {
            "action_type": "resolve",
            "resolved_ticket_id": tid,
            "resolution_note": "Resolved by fallback. Escalated to tier3 for review.",
        }
    return {"action_type": "classify", "category": "general"}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str, seed: int) -> Tuple[float, Dict]:
    """
    Run one full episode for the given task.
    Returns (final_reward, breakdown).
    """
    print(f"\n{'='*60}")
    print(f"TASK: {task_id.upper()} | SEED: {seed}")
    print(f"{'='*60}")

    # Reset environment
    reset_resp = env_reset(task_id, seed)
    obs = reset_resp["observation"]
    max_steps = reset_resp.get("max_steps", MAX_STEPS)

    print(f"  Max steps: {max_steps}")
    if obs.get("tickets"):
        first = obs["tickets"][0]
        print(f"  Ticket: [{first['ticket_id']}] {first['subject'][:60]}")

    history: List[str] = []
    final_reward: float = 0.0
    final_breakdown: Dict = {}

    for step_num in range(1, max_steps + 1):
        # Check if already done
        if obs.get("resolved_tickets") and task_id == "easy":
            print(f"  Episode complete at step {step_num - 1}.")
            break

        user_prompt = build_user_prompt(obs, history)
        raw_response = call_llm(user_prompt)
        action = parse_action(raw_response)

        if action is None:
            print(f"  Step {step_num}: Could not parse action from LLM. Using fallback.")
            action = fallback_action(obs)

        print(f"  Step {step_num}: {action.get('action_type', '?')} → ", end="")

        try:
            result = env_step(action)
        except Exception as exc:
            print(f"ERROR calling env: {exc}")
            break

        obs = result["observation"]
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        info = result.get("info", {})
        step_reward = info.get("step_reward", 0.0)
        breakdown = info.get("breakdown", {})
        error = info.get("error")

        final_reward = reward
        final_breakdown = breakdown

        status = f"reward={reward:.3f} (+{step_reward:.3f})"
        if error:
            status += f" | ERROR: {error}"
        print(status)

        history.append(
            f"Step {step_num}: {action.get('action_type', '?')} → "
            f"reward={reward:.3f} (+{step_reward:.3f})"
        )

        if done:
            print(f"  Episode complete at step {step_num}.")
            break

    return final_reward, final_breakdown


# ---------------------------------------------------------------------------
# Main — run all 3 tasks
# ---------------------------------------------------------------------------

def main() -> None:
    # Verify server is up
    try:
        health = _get("/health")
        print(f"Environment server: {ENV_HTTP_URL} — status: {health.get('status', '?')}")
    except Exception as exc:
        print(
            f"ERROR: Cannot connect to environment server at {ENV_HTTP_URL}.\n"
            f"  Reason: {exc}\n"
            f"  Start the server with: uvicorn server:app --host 0.0.0.0 --port 8000",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nModel: {MODEL_NAME}")
    print(f"Seed:  {SEED}")

    results: Dict[str, float] = {}

    for task_id in TASKS:
        reward, breakdown = run_episode(task_id, seed=SEED)
        results[task_id] = reward
        print(f"\n  Final reward ({task_id}): {reward:.4f}")
        print(f"  Breakdown: {breakdown}")

    # Final summary
    print(f"\n{'='*60}")
    print("BASELINE SCORES")
    print(f"{'='*60}")
    for task_id in TASKS:
        score = results[task_id]
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_id:<8} [{bar}] {score:.4f}")
    overall = sum(results.values()) / len(results)
    print(f"{'─'*60}")
    print(f"  {'AVERAGE':<8}              {overall:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
