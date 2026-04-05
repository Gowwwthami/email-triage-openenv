"""
Email Triage Hackathon API
FastAPI server for email classification, extraction, and reply generation.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI


# Pydantic models for request/response
class ClassifyRequest(BaseModel):
    email: str


class ClassifyResponse(BaseModel):
    category: str


class ExtractRequest(BaseModel):
    email: str


class ExtractResponse(BaseModel):
    order_id: str
    issue: str
    intent: str


class SuggestRequest(BaseModel):
    email: str
    category: str


class SuggestResponse(BaseModel):
    response: str


# Initialize FastAPI app
app = FastAPI(
    title="Email Triage Hackathon API",
    description="API for email classification, information extraction, and reply generation",
    version="1.0.0"
)


def get_openai_client() -> OpenAI | None:
    """Initialize OpenAI client from environment variables."""
    api_base_url = os.getenv("API_BASE_URL")
    hf_token = os.getenv("HF_TOKEN")
    
    if not api_base_url:
        return None
    
    return OpenAI(
        base_url=api_base_url,
        api_key=hf_token or ""
    )


def get_model_name() -> str:
    """Get model name from environment."""
    return os.getenv("MODEL_NAME", "gpt-4o-mini")


@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint - API info."""
    return {
        "message": "Email Triage Hackathon API is running",
        "endpoints": {
            "classify": "POST /classify",
            "extract": "POST /extract",
            "suggest": "POST /suggest"
        }
    }


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest) -> Dict[str, str]:
    """
    Classify email into category.
    
    Input: {"email": "text"}
    Output: {"category": "refund"}
    """
    client = get_openai_client()
    model_name = get_model_name()
    
    if not client:
        # Fallback to rule-based classification
        category = rule_based_classify(payload.email)
        return {"category": category}
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": """You are an email classification assistant. 
Classify the email into exactly one of these categories: refund, billing, technical, sales, account, complaint, shipping, other.
Respond with ONLY the category name, nothing else."""
                },
                {"role": "user", "content": f"Classify this email:\n\n{payload.email}"}
            ]
        )
        
        category = response.choices[0].message.content.strip().lower()
        # Validate category
        valid_categories = ["refund", "billing", "technical", "sales", "account", "complaint", "shipping", "other"]
        if category not in valid_categories:
            category = "other"
        
        return {"category": category}
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(exc)}")


@app.post("/extract", response_model=ExtractResponse)
def extract(payload: ExtractRequest) -> Dict[str, str]:
    """
    Extract structured information from email.
    
    Input: {"email": "text"}
    Output: {"order_id": "...", "issue": "...", "intent": "..."}
    """
    client = get_openai_client()
    model_name = get_model_name()
    
    if not client:
        # Fallback to rule-based extraction
        return rule_based_extract(payload.email)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": """You are an information extraction assistant.
Extract the following from the email and return ONLY a JSON object with these exact keys:
- order_id: The order ID mentioned (or "unknown" if not found)
- issue: Brief description of the problem (max 10 words)
- intent: The customer's intent (e.g., "request_refund", "ask_question", "report_issue")

Return valid JSON only, no markdown formatting."""
                },
                {"role": "user", "content": f"Extract information from this email:\n\n{payload.email}"}
            ]
        )
        
        content = response.choices[0].message.content.strip()
        # Try to parse JSON, fallback to defaults if fails
        import json
        try:
            result = json.loads(content)
            return {
                "order_id": result.get("order_id", "unknown"),
                "issue": result.get("issue", "unknown"),
                "intent": result.get("intent", "unknown")
            }
        except json.JSONDecodeError:
            return {"order_id": "unknown", "issue": content[:50], "intent": "unknown"}
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(exc)}")


@app.post("/suggest", response_model=SuggestResponse)
def suggest(payload: SuggestRequest) -> Dict[str, str]:
    """
    Generate professional reply to email.
    
    Input: {"email": "...", "category": "..."}
    Output: {"response": "..."}
    """
    client = get_openai_client()
    model_name = get_model_name()
    
    if not client:
        # Fallback to template-based response
        response_text = template_based_suggest(payload.email, payload.category)
        return {"response": response_text}
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional customer support agent.
Write a polite, helpful, and concise email response.
Address the customer's concern professionally.
Keep the response under 150 words."""
                },
                {
                    "role": "user",
                    "content": f"Category: {payload.category}\n\nCustomer email:\n{payload.email}\n\nWrite a professional response:"
                }
            ]
        )
        
        response_text = response.choices[0].message.content.strip()
        return {"response": response_text}
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {str(exc)}")


# Rule-based fallback functions
def rule_based_classify(email: str) -> str:
    """Rule-based classification when LLM is not available."""
    text = email.lower()
    
    if any(word in text for word in ["refund", "money back", "return"]):
        return "refund"
    elif any(word in text for word in ["invoice", "charge", "billing", "payment"]):
        return "billing"
    elif any(word in text for word in ["crash", "error", "bug", "not working"]):
        return "technical"
    elif any(word in text for word in ["price", "quote", "purchase", "buy"]):
        return "sales"
    elif any(word in text for word in ["login", "password", "account", "access"]):
        return "account"
    elif any(word in text for word in ["complaint", "unhappy", "terrible", "rude"]):
        return "complaint"
    elif any(word in text for word in ["shipping", "delivery", "package", "order status"]):
        return "shipping"
    else:
        return "other"


def rule_based_extract(email: str) -> Dict[str, str]:
    """Rule-based extraction when LLM is not available."""
    text = email.lower()
    
    # Extract order ID (simple pattern matching)
    import re
    order_match = re.search(r'order\s*(?:#|number|id)?\s*:?\s*(\d+|[A-Z0-9-]+)', text, re.IGNORECASE)
    order_id = order_match.group(1) if order_match else "unknown"
    
    # Determine issue
    if "refund" in text:
        issue = "refund request"
    elif "not working" in text or "crash" in text:
        issue = "technical problem"
    elif "delivery" in text or "shipping" in text:
        issue = "shipping issue"
    else:
        issue = "general inquiry"
    
    # Determine intent
    if "refund" in text or "money back" in text:
        intent = "request_refund"
    elif "help" in text or "support" in text:
        intent = "request_help"
    elif "question" in text:
        intent = "ask_question"
    else:
        intent = "report_issue"
    
    return {"order_id": order_id, "issue": issue, "intent": intent}


def template_based_suggest(email: str, category: str) -> str:
    """Template-based response when LLM is not available."""
    templates = {
        "refund": "Thank you for contacting us about your refund request. We have received your inquiry and are reviewing your order. Our team will process your refund within 3-5 business days. You will receive a confirmation email once completed.",
        "billing": "Thank you for reaching out regarding your billing inquiry. We have reviewed your account and will investigate this matter. Our billing team will contact you within 24 hours with a resolution.",
        "technical": "Thank you for reporting this technical issue. Our engineering team has been notified and is investigating. We will provide an update within 24 hours. In the meantime, please try clearing your browser cache.",
        "sales": "Thank you for your interest in our products. We have received your inquiry and one of our sales representatives will contact you within 24 hours to discuss your requirements and provide a customized quote.",
        "account": "Thank you for contacting us about your account. We have received your request and our support team is reviewing it. We will send you instructions to resolve this issue within the next few hours.",
        "complaint": "We sincerely apologize for the inconvenience you experienced. Your complaint has been escalated to our customer relations team. A specialist will contact you within 24 hours to resolve this matter.",
        "shipping": "Thank you for contacting us about your shipment. We are tracking your package and will provide an update on its status within 24 hours. We appreciate your patience.",
        "other": "Thank you for contacting us. We have received your message and will respond within 24 hours. If this is urgent, please call our support line."
    }
    
    return templates.get(category, templates["other"])


def main() -> None:
    """Main entry point for running the server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
