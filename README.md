---
title: Customer Support Ticket Resolution Environment Server
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Customer Support Ticket Resolution Environment

An **OpenEnv-compatible reinforcement learning environment** where an AI agent resolves customer support tickets by classifying, prioritizing, responding, escalating, and closing them according to support policies.

## Motivation

Customer support ticket resolution is a high-value real-world task performed by millions of agents (human and AI) daily. Getting it right requires:
- **Classification** — understanding what kind of problem the customer has
- **Prioritization** — routing by urgency and customer tier
- **Policy lookup** — knowing the rules before acting
- **Escalation judgment** — knowing when to involve specialized teams
- **Communication** — writing professional, empathetic responses

This environment provides a structured, graded benchmark for training and evaluating agents on this workflow.

---

## Tasks

### Task 1 — Easy
**Classify and respond to a single ticket**

The agent receives one support ticket and must:
1. `classify` — assign the correct category
2. `set_priority` — assign the correct priority
3. `respond` — write a customer-facing response
4. `resolve` — close the ticket with an internal note

| Sub-reward | Points | Condition |
|---|---|---|
| classify_correct | 0.30 | Category matches ground truth |
| priority_correct | 0.20 | Priority matches ground truth |
| response_sent | 0.20 | A response was written |
| response_quality | 0.20 | Response contains required keywords |
| efficiency_bonus | 0.10 | Completed in ≤ 4 steps |
| **Max** | **1.00** | |

### Task 2 — Medium
**Triage a queue of 5 tickets**

The agent receives 5 tickets spanning different tiers and categories. It must submit one `bulk_triage` action assigning correct category and priority to each.

| Sub-reward | Points | Condition |
|---|---|---|
| triage_submitted | 0.10 | bulk_triage action taken |
| category_accuracy | 0.40 | Per-ticket category accuracy (proportional) |
| priority_accuracy | 0.40 | Per-ticket priority accuracy (proportional) |
| efficiency_bonus | 0.10 | Completed in ≤ 3 steps |
| **Max** | **1.00** | |

### Task 3 — Hard
**Resolve a complex enterprise billing dispute**

The agent handles an enterprise customer with a billing dispute and multiple prior contacts. Must follow the full resolution procedure:

| Sub-reward | Points | Condition |
|---|---|---|
| classify_correct | 0.15 | Category = "billing" |
| priority_correct | 0.15 | Priority = "critical" |
| policy_looked_up | 0.15 | lookup_policy action taken |
| correct_policy_found | 0.10 | Correct policy retrieved |
| escalated | 0.15 | escalate action taken |
| correct_escalation_tier | 0.10 | Escalation to "tier3" |
| resolved | 0.10 | resolve action taken |
| resolution_quality | 0.10 | Resolution note contains required keywords |
| **Max** | **1.00** | |

---

## Action Space

All actions are JSON objects with an `action_type` field:

```json
// Classify ticket category
{"action_type": "classify", "category": "billing|technical|account|feature_request|abuse|general"}

// Assign priority
{"action_type": "set_priority", "priority": "critical|high|medium|low"}

// Send customer response
{"action_type": "respond", "response_text": "Dear customer, ..."}

// Escalate to support tier
{"action_type": "escalate", "escalation_tier": "tier2|tier3|legal|executive", "escalation_reason": "..."}

// Search policy knowledge base
{"action_type": "lookup_policy", "policy_query": "enterprise billing refund"}

// Close ticket
{"action_type": "resolve", "resolved_ticket_id": "TKT-001", "resolution_note": "..."}

// Triage multiple tickets at once (medium task only)
{
  "action_type": "bulk_triage",
  "triage_list": [
    {"ticket_id": "TKT-001", "category": "billing", "priority": "high"},
    ...
  ]
}
```

### Reward & Penalty

- **Partial credit**: each correct sub-step earns reward immediately (non-sparse)
- **Invalid action penalty**: `-0.05` per invalid/malformed action (floor = 0.0)
- **Efficiency bonus**: +0.10 for completing within the step threshold
- **Step limit**: episode ends at max_steps — wasted steps reduce score via lost efficiency bonus

### Priority Matrix

| Tier | billing | technical | account | feature_request | general |
|---|---|---|---|---|---|
| enterprise | critical | high | high | low | medium |
| pro | high | medium | medium | low | medium |
| free | medium | low | low | low | low |

> **Abuse is always critical** regardless of customer tier.

---

## Observation Space

Each step returns an `Observation` JSON object:

| Field | Type | Description |
|---|---|---|
| `step` | int | Current step (0-indexed) |
| `task_id` | string | "easy" / "medium" / "hard" |
| `tickets` | array | Support tickets (1 or 5 depending on task) |
| `active_ticket_id` | string\|null | Which ticket to focus on (hard task) |
| `policy_result` | object\|null | Result of last `lookup_policy` action |
| `action_feedback` | string\|null | Human-readable feedback on last action |
| `action_valid` | bool\|null | Whether last action was structurally valid |
| `escalation_queue` | array | Ticket IDs currently escalated |
| `resolved_tickets` | array | Ticket IDs that have been resolved |
| `instructions` | string | Full task instructions |

Each `Ticket` object contains:
- `ticket_id`, `subject`, `body`
- `customer_tier`: free / pro / enterprise
- `channel`: email / chat / phone
- `previous_contacts`: integer
- `account_age_days`: integer

---

## Setup & Usage

### Local (without Docker)

```bash
# Install dependencies
pip install -e ".[inference]"

# Start server
uvicorn server:app --host 0.0.0.0 --port 8000

# In another terminal — run inference
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-id"
export HF_TOKEN="your-token"
python inference.py
```

### Docker

```bash
# Build
docker build -t customer-support-env:latest .

# Run server
docker run -p 8000:8000 customer-support-env:latest

# Run inference (separate terminal)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-id"
export HF_TOKEN="your-token"
export ENV_HTTP_URL="http://localhost:8000"
python inference.py
```

### Hugging Face Spaces

The environment is deployed as an HF Space. After the Space starts:

```bash
export ENV_HTTP_URL="https://your-username-customer-support-env.hf.space"
python inference.py
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/info` | openenv.yaml metadata |
| GET | `/health` | Runtime status |
| POST | `/reset` | Start new episode |
| POST | `/step` | Execute one action |
| GET | `/state` | Current observation (no side effects) |
| WS | `/ws` | WebSocket interface |

### Reset

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'
```

### Step

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "classify", "category": "billing"}}'
```

---

## Baseline Scores

Tested with `seed=42`, model: `meta-llama/Llama-3.1-8B-Instruct` via HF Inference Router:

| Task | Score | Breakdown |
|---|---|---|
| easy | 0.20 | Response sent (0.20); missed correct classify/priority |
| medium | 0.92 | Near-perfect triage; strong category+priority accuracy |
| hard | 0.90 | Full workflow: classify→priority→policy→escalate→resolve |
| **Average** | **0.67** | |

> Scores vary by model. Larger models score higher on the easy task (requires precise classification). The hard task rewards stepwise procedure — models that follow instructions do well.

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | Yes (inference) | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | Yes (inference) | — | Model identifier |
| `OPENAI_API_KEY` | Yes (inference) | — | OpenAI / HF API key (alias: `HF_TOKEN`) |
| `HF_TOKEN` | Yes (inference) | — | Hugging Face API key (alias: `OPENAI_API_KEY`) |
| `ENV_HTTP_URL` | No | `http://localhost:8000` | Environment server URL |
| `TASK_ID` | No | `easy` | Default task for server |
| `SEED` | No | `42` | Random seed for reproducibility |

---

## Project Structure

```
.
├── customer_support_env/       # Python package
│   ├── __init__.py
│   ├── env.py                  # Core environment class
│   ├── models.py               # Pydantic models
│   ├── tasks.py                # Task specs and scenario generation
│   ├── graders.py              # Reward/scoring functions
│   ├── policies.py             # Policy knowledge base lookup
│   └── data/
│       ├── tickets.json        # 30 support ticket corpus
│       └── policy_kb.json      # 8 support policy documents
├── server.py                   # FastAPI server
├── inference.py                # Baseline inference script
├── openenv.yaml                # OpenEnv metadata
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── tests/
    └── test_env.py
```
