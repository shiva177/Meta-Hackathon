"""
Inference Script — Customer Support Ticket Resolution Environment

Required environment variables:
  API_BASE_URL  : API endpoint for the LLM (default: https://api.openai.com/v1)
  MODEL_NAME    : Model identifier (default: gpt-4o-mini)
  HF_TOKEN      : Hugging Face API token (mandatory, no default)

Optional:
  ENV_HTTP_URL  : Environment server URL (default: http://localhost:8000)
  SEED          : Random seed for reproducibility (default: 42)

Output format (per guidelines):
  [START] task=<task> env=customer-support-ticket-resolution model=<model>
  [STEP]  step=<n> action=<action_type> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Usage:
  python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — with required defaults per guidelines
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
ENV_HTTP_URL: str = os.environ.get("ENV_HTTP_URL", "http://localhost:8000")
SEED: int = int(os.environ.get("SEED", "42"))

ENV_NAME: str = "customer-support-ticket-resolution"
TASKS: List[str] = ["easy", "medium", "hard"]
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 512

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Environment HTTP helpers
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


# ---------------------------------------------------------------------------
# Prompts
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
    instructions = obs.get("instructions", "")
    action_feedback = obs.get("action_feedback")
    action_valid = obs.get("action_valid")
    policy_result = obs.get("policy_result")
    escalation_queue = obs.get("escalation_queue", [])
    resolved_tickets = obs.get("resolved_tickets", [])

    lines = [instructions, "", "--- TICKETS ---"]
    for i, ticket in enumerate(tickets, 1):
        lines.append(f"[Ticket {i}]")
        lines.append(_format_ticket(ticket))
        lines.append("")

    if action_feedback:
        valid_str = "OK" if action_valid else "INVALID"
        lines.append(f"--- LAST ACTION [{valid_str}] ---")
        lines.append(action_feedback)
        lines.append("")

    if policy_result and policy_result.get("matched_policy_id"):
        lines.append("--- POLICY LOOKUP RESULT ---")
        lines.append(f"  Policy: [{policy_result['matched_policy_id']}] {policy_result.get('policy_title', '')}")
        body = policy_result.get("policy_body", "")
        if body:
            lines.append(f"  Content: {body[:400]}")
        lines.append("")

    if escalation_queue:
        lines.append(f"Escalated: {escalation_queue}")
    if resolved_tickets:
        lines.append(f"Resolved: {resolved_tickets}")

    if history:
        lines.append("")
        lines.append("--- HISTORY ---")
        lines.extend(history[-5:])

    lines.append("")
    lines.append("Your next action (JSON only):")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call + action parsing
# ---------------------------------------------------------------------------

def call_llm(user_prompt: str) -> str:
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
        return f"__ERROR__:{exc}"


def parse_action(text: str) -> Optional[Dict]:
    text = text.strip()
    if not text or text.startswith("__ERROR__"):
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict) and "action_type" in obj:
                return obj
        except json.JSONDecodeError:
            pass
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
    task_id = obs.get("task_id", "easy")
    if task_id == "medium":
        tickets = obs.get("tickets", [])
        return {
            "action_type": "bulk_triage",
            "triage_list": [
                {"ticket_id": t["ticket_id"], "category": "general", "priority": "medium"}
                for t in tickets
            ],
        }
    if task_id == "hard":
        tickets = obs.get("tickets", [])
        tid = tickets[0]["ticket_id"] if tickets else "TKT-001"
        return {
            "action_type": "resolve",
            "resolved_ticket_id": tid,
            "resolution_note": "Credit note issued. Escalated to tier3. Account manager assigned.",
        }
    return {"action_type": "classify", "category": "general"}


# ---------------------------------------------------------------------------
# Episode runner — emits [START] / [STEP] / [END] per guidelines
# ---------------------------------------------------------------------------

def run_episode(task_id: str, seed: int) -> Tuple[float, int, List[float], bool]:
    """
    Run one full episode. Returns (final_reward, steps_taken, step_rewards, success).
    Emits [START], [STEP], [END] lines to stdout.
    """
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    step_rewards: List[float] = []
    steps_taken: int = 0
    success: bool = False
    last_error: Optional[str] = None

    try:
        reset_resp = env_reset(task_id, seed)
        obs = reset_resp["observation"]
        max_steps = reset_resp.get("max_steps", 10)

        history: List[str] = []

        for step_num in range(1, max_steps + 1):
            steps_taken = step_num
            user_prompt = build_user_prompt(obs, history)
            raw_response = call_llm(user_prompt)
            action = parse_action(raw_response)

            if action is None:
                action = fallback_action(obs)

            action_str = action.get("action_type", "unknown")

            try:
                result = env_step(action)
            except Exception as exc:
                last_error = str(exc)
                print(
                    f"[STEP] step={step_num} action={action_str} reward=0.01 "
                    f"done=false error={last_error}",
                    flush=True,
                )
                break

            obs = result["observation"]
            reward = max(0.01, min(0.99, float(result.get("reward", 0.01))))
            done = result.get("done", False)
            info = result.get("info", {})
            last_error = info.get("error") or None
            step_rewards.append(reward)  # cumulative reward per step (matches [STEP] reward= values)

            error_str = last_error if last_error else "null"
            done_str = "true" if done else "false"
            print(
                f"[STEP] step={step_num} action={action_str} reward={reward:.2f} "
                f"done={done_str} error={error_str}",
                flush=True,
            )

            history.append(f"step={step_num} action={action_str} reward={reward:.2f}")

            if done:
                success = True
                break

    except Exception as exc:
        last_error = str(exc)

    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards) if step_rewards else "0.01"
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={steps_taken} rewards={rewards_str}",
        flush=True,
    )

    final_reward = float(step_rewards[-1]) if step_rewards else 0.01
    return final_reward, steps_taken, step_rewards, success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Verify server is reachable
    try:
        health = _get("/health")
        assert health.get("status") == "ok"
    except Exception as exc:
        print(f"ERROR: Cannot connect to environment at {ENV_HTTP_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    all_rewards: Dict[str, float] = {}

    for task_id in TASKS:
        cumulative, steps, step_rewards, success = run_episode(task_id, seed=SEED)
        all_rewards[task_id] = cumulative
        print(flush=True)

    # Summary
    print("=" * 60, flush=True)
    print("BASELINE SCORES", flush=True)
    print("=" * 60, flush=True)
    for task_id in TASKS:
        score = all_rewards[task_id]
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_id:<8} [{bar}] {score:.2f}", flush=True)
    overall = sum(all_rewards.values()) / len(all_rewards)
    print(f"{'─'*60}", flush=True)
    print(f"  {'AVERAGE':<8}              {overall:.2f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
