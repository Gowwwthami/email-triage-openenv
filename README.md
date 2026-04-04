# Email Triage OpenEnv Environment

A realistic Email Triage OpenEnv environment where an AI agent must manage an inbox by classifying emails, prioritizing urgent emails, taking appropriate actions, and generating professional replies.

## Overview

This environment simulates a real-world customer support email workflow where an AI agent:
- **Classifies** incoming emails into categories (billing, technical, sales, account, complaint, shipping, other)
- **Prioritizes** emails based on urgency (low, medium, high, urgent)
- **Takes actions** (reply, escalate, archive)
- **Selects appropriate reply templates** for professional responses

The environment provides step-wise rewards based on:
- Category classification accuracy (+0.3)
- Priority handling (+0.2)
- Action correctness (+0.2)
- Reply template selection (+0.3)

## Key Features

- **Typed Pydantic models** for observation, action, state, and reward
- **Gym-compatible loop**: `reset()`, `step(action)`, `state()`
- **Deterministic rule-based grading** with normalized scores in `[0.0, 1.0]`
- **Three progressively harder tasks** evaluating different agent capabilities
- **Enhanced state tracking**: emails processed, replies sent, escalations, urgent handled
- **Synthetic dataset** of 30 fully labeled support emails
- **Hybrid agent architecture**: Rules for classification, deterministic templates for replies

## Project Structure

```text
email-triage-env/
├── inference.py
├── pyproject.toml
├── uv.lock
├── openenv.yaml
├── Dockerfile
├── README.md
├── requirements.txt
├── server/
│   ├── __init__.py
│   ├── app.py
├── src/
│   ├── env.py
│   ├── models.py
│   ├── dataset.py
│   ├── tasks.py
│   ├── rewards.py
│   ├── graders.py
```

## Environment Design

### Observation Space

Each step provides:
- `email_id`: unique email id
- `email_text`: raw email body
- `task_id`: active task identifier
- `valid_categories`, `valid_priorities`, `valid_actions`

### Action Space

Agent action (`Action` model):
- `category`: one of `billing|technical|sales|account|complaint|shipping|other`
- `priority`: one of `low|medium|high|urgent`
- `action`: one of `reply|escalate|archive`
- `reply_template`: template key string

### State Space

`state()` returns:
- `task_id`: active task identifier
- `current_index`: position in email queue
- `total_emails`: total emails in dataset
- `cumulative_reward`: total reward accumulated
- `last_reward`: reward from last step
- `done`: episode completion flag

Additionally, the environment tracks enhanced statistics in `info["stats"]`:
- `emails_processed`: count of processed emails
- `emails_remaining`: emails left to process
- `replies_sent`: number of reply actions taken
- `escalations`: number of escalate actions
- `archived`: number of archive actions
- `urgent_handled`: number of urgent priority emails processed

## Tasks

| Task | Difficulty | Required Outputs | Description |
|------|------------|------------------|-------------|
| `task_easy` | Easy | category | Email category classification only |
| `task_medium` | Medium | category, priority, action | Category + priority assignment + action selection |
| `task_hard` | Hard | category, priority, action, reply_template | Full email triage with reply template selection |

### Task Logic

- **task_easy**: Only category matters. Other fields use dummy values.
- **task_medium**: Category, priority, and action must be correct. Reply template can be generic.
- **task_hard**: All fields must be correct including the exact reply template mapped from category.

## Reward Function

Continuous per-step reward with partial credit:

- Correct category: `+0.3`
- Correct priority: `+0.2`
- Correct action: `+0.2`
- Correct reply template: `+0.3`

Penalties:
- Wrong classification (category mismatch): `-0.2`
- Unnecessary escalation: `-0.3`

Rewards are computed at every step, not only at episode end.

## Agent Architecture

The inference agent uses a **deterministic rule-based approach**:

| Component | Method | Logic |
|-----------|--------|-------|
| Category | Rule-based | Keyword matching with prioritized categories |
| Priority | Rule-based | Urgency keyword detection |
| Action | Rule-based | Category + priority based logic |
| Reply Template | Rule-mapped | Direct mapping from category to template |

### Category Detection Priority

1. Complaint (emotional indicators)
2. Billing (financial indicators)
3. Shipping (delivery indicators)
4. Account (login/access indicators)
5. Sales (pricing/business indicators)
6. Technical (problem indicators)
7. Other (default)

### Reply Template Mapping

```
billing → billing_reply / billing_refund / billing_invoice
technical → technical_reply / tech_troubleshoot / escalate_specialist
sales → sales_reply / sales_pricing
account → account_reply / account_unlock
complaint → complaint_reply / complaint_apology / escalate_specialist
shipping → shipping_reply / shipping_update
other → general_reply / archive_no_reply
```

## Baseline Scores

| Task | Score | Category Accuracy | Priority Accuracy | Action Accuracy | Reply Accuracy |
|------|-------|-------------------|-------------------|-----------------|----------------|
| Easy | 0.9333 | 93.33% | - | - | - |
| Medium | 0.8500 | 93.33% | 76.67% | 86.67% | - |
| Hard | 0.8500 | 93.33% | 76.67% | 86.67% | 83.33% |

## Grader

Deterministic scoring in `src/graders.py`:

- Final score is normalized: `correct_decisions / total_possible_decisions`
- Always in `[0.0, 1.0]`
- No external LLM used for grading

## Setup

### Local Python

```bash
pip install -r requirements.txt
python inference.py
```

Optional env vars for OpenAI-compatible endpoint:

```bash
export API_BASE_URL="https://your-openai-compatible-endpoint/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_token"
python inference.py
```

If `API_BASE_URL` is not set, inference uses deterministic local heuristics.

### Run the API Server (app.py)

The OpenEnv validator and Hugging Face Space health checks require a live server that responds to `POST /reset`.

From repo root:

```bash
python -m server.app
```

Alternative command from the `server/` directory:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Quick endpoint checks:

```bash
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d '{}'
curl http://127.0.0.1:7860/state
```

### Docker

```bash
docker build -t email-triage-env .
docker run --rm \
  -p 7860:7860 \
  -e API_BASE_URL="https://your-openai-compatible-endpoint/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="your_token" \
  email-triage-env
```

It also runs without external endpoint:

```bash
docker run --rm -p 7860:7860 email-triage-env
```

Current Docker behavior:

- Container starts the FastAPI server (`python -m server.app`)
- Server binds to port `7860`
- Required for Hugging Face `/reset` health checks

### Hugging Face Spaces Deployment

This environment is ready for deployment to Hugging Face Spaces:

1. Create a new Space on Hugging Face with **Docker** SDK.
2. Add the `openenv` tag to your Space.
3. Push this repository to the Space.
4. Set environment variables in Space Settings -> Variables/Secrets:
   - `API_BASE_URL` (optional - uses heuristics if not set)
   - `MODEL_NAME` (default: gpt-4o-mini)
   - `HF_TOKEN` (for API access)
5. Wait for build/startup and verify endpoint:

```bash
curl -X POST https://<your-space>.hf.space/reset -H "Content-Type: application/json" -d '{}'
```

Expected: HTTP `200`.

6. Run validator before submission:

```bash
openenv validate
```

And run the submission script from the requirement guide:

```bash
./scripts/validate-submission.sh https://<your-space>.hf.space .
```

The Space now starts the API server on startup, not `inference.py`.

## How Inference Works

`inference.py`:
- Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- Creates OpenAI client if endpoint is configured
- Runs all 3 tasks sequentially
- Emits strict structured logs:
  - `[START] ...`
  - `[STEP] ...`
  - `[END] ...`

## Example Output

### Heuristic Baseline (No API)

```text
[START] task_id=task_easy model_name=gpt-4o-mini api_enabled=0 total_steps=30
[STEP] task_id=task_easy step=01 email_id=E001 reward=0.3000 cumulative_reward=0.3000 category=billing priority=medium action=reply reply_template=general_reply
...
[END] task_id=task_easy steps=30 final_score=0.9333 cumulative_reward=8.0000 avg_reward=0.2667 category_accuracy=0.9333 priority_accuracy=0.3000 action_accuracy=0.5667 reply_accuracy=0.0000
```

### Baseline Scores

| Task | Difficulty | Heuristic Score | Expected LLM Score |
|------|------------|-----------------|-------------------|
| task_easy | Easy | 0.8333 | 0.90+ |
| task_medium | Medium | 0.7667 | 0.85+ |
| task_hard | Hard | 0.7667 | 0.80+ |
