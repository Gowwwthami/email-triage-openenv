---
title: Email Triage Hackathon API
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# Email Triage Hackathon API

A FastAPI-based email triage system for hackathon evaluation. The API provides three endpoints for email classification, information extraction, and reply generation.

## Overview

This project implements a 3-step email processing pipeline:

1. **Classify** → Categorize emails (refund, billing, technical, sales, account, complaint, shipping, other)
2. **Extract** → Extract structured information (order_id, issue, intent)
3. **Suggest** → Generate professional email replies

## API Endpoints

### POST /classify
Classify an email into a category.

**Input:**
```json
{
  "email": "I want a refund for order #12345"
}
```

**Output:**
```json
{
  "category": "refund"
}
```

### POST /extract
Extract structured information from an email.

**Input:**
```json
{
  "email": "My order #12345 arrived damaged. I need a refund."
}
```

**Output:**
```json
{
  "order_id": "12345",
  "issue": "damaged product",
  "intent": "request_refund"
}
```

### POST /suggest
Generate a professional reply to an email.

**Input:**
```json
{
  "email": "I want a refund for order #12345",
  "category": "refund"
}
```

**Output:**
```json
{
  "response": "Thank you for contacting us about your refund request..."
}
```

## Project Structure

```
email-triage-hackathon/
├── app.py              # FastAPI server with API endpoints
├── inference.py        # Inference script with structured logging
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
└── README.md          # Project documentation
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | Base URL for OpenAI-compatible API | Optional |
| `MODEL_NAME` | LLM model name (default: gpt-4o-mini) | Optional |
| `HF_TOKEN` | API token for authentication | Optional |

## Setup Instructions

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
python app.py
```

3. In another terminal, run inference:
```bash
python inference.py
```

### Docker

```bash
docker build -t email-triage-api .
docker run -p 7860:7860 email-triage-api
```

### Hugging Face Spaces

1. Create a new Space with Docker SDK
2. Push this repository to the Space
3. Add environment variables in Space Settings if using LLM
4. The API will be available at `https://<your-space>.hf.space`

## Inference Script

The `inference.py` script demonstrates the complete pipeline:

```bash
# Run with sample email
python inference.py

# Run with custom email
python inference.py "Your email text here"

# Run with email from file
python inference.py path/to/email.txt
```

### Expected Output

```
[START] task=email-triage env=openenv model=gpt-4o-mini
[STEP] step=1 action=classify_email reward=0.33 done=false error=null
[STEP] step=2 action=extract_entities reward=0.33 done=false error=null
[STEP] step=3 action=generate_reply reward=0.34 done=true error=null
[END] success=true steps=3 score=1.00 rewards=0.33,0.33,0.34
```

## API Testing

Test the endpoints with curl:

```bash
# Classify
curl -X POST http://localhost:7860/classify \
  -H "Content-Type: application/json" \
  -d '{"email": "I need a refund for order #123"}'

# Extract
curl -X POST http://localhost:7860/extract \
  -H "Content-Type: application/json" \
  -d '{"email": "My order #123 arrived damaged"}'

# Suggest
curl -X POST http://localhost:7860/suggest \
  -H "Content-Type: application/json" \
  -d '{"email": "I need a refund", "category": "refund"}'
```

## Fallback Behavior

If `API_BASE_URL` is not set, the system uses rule-based fallbacks:
- **Classification**: Keyword matching
- **Extraction**: Regex pattern matching
- **Suggestion**: Template-based responses

## License

MIT
