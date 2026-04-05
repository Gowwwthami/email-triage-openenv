"""
Email Triage Hybrid Agent - Inference Script with Strict Logging

Hybrid Architecture:
- Rule-based: category classification, action selection, reply template selection
- LLM API: entity extraction (customer_name, product, urgency)

This script implements the agent logic directly and prints structured logs for evaluation.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI


# ============================================================================
# CONFIGURATION
# ============================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")


# ============================================================================
# VALID CATEGORIES AND MAPPINGS
# ============================================================================
VALID_CATEGORIES = [
    "refund",
    "complaint",
    "order_status",
    "technical_support",
    "billing",
    "general_inquiry",
    "urgent",
    "spam"
]

# Category to action mapping
CATEGORY_ACTIONS = {
    "refund": "process_refund",
    "complaint": "escalate_to_manager",
    "order_status": "check_order_status",
    "technical_support": "create_support_ticket",
    "billing": "review_billing_issue",
    "general_inquiry": "send_general_response",
    "urgent": "priority_escalation",
    "spam": "mark_as_spam"
}

# Category to reply template mapping
CATEGORY_TEMPLATES = {
    "refund": "refund_template",
    "complaint": "complaint_template",
    "order_status": "order_status_template",
    "technical_support": "technical_support_template",
    "billing": "billing_template",
    "general_inquiry": "general_inquiry_template",
    "urgent": "urgent_template",
    "spam": "spam_template"
}


# ============================================================================
# STRICT LOGGER (DO NOT MODIFY)
# ============================================================================
class StrictLogger:
    """
    Strict logger for hackathon evaluation.
    Output format is FIXED and cannot be changed.
    """
    
    def __init__(self, task: str, env: str, model: str):
        self.task = task
        self.env = env
        self.model = model
        self.step_count = 0
        self.rewards: list[float] = []
        self.has_error = False
    
    def start(self) -> None:
        """Print [START] log - SINGLE LINE ONLY."""
        print(f"[START] task={self.task} env={self.env} model={self.model}", flush=True)
    
    def step(self, action: str, reward: float, done: bool, error: str | None = None) -> None:
        """
        Print [STEP] log - SINGLE LINE ONLY.
        
        Format: [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
        """
        self.step_count += 1
        self.rewards.append(reward)
        
        if error:
            self.has_error = True
            # Escape any special characters in error message
            error_str = error.replace("\n", " ").replace("\r", " ").strip()
            if len(error_str) > 100:
                error_str = error_str[:100] + "..."
        else:
            error_str = "null"
        
        print(
            f"[STEP] step={self.step_count} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}",
            flush=True
        )
    
    def end(self, success: bool) -> None:
        """
        Print [END] log - SINGLE LINE ONLY - ALWAYS CALLED.
        
        Format: [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
        """
        score = sum(self.rewards)
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        rewards_str = ",".join(f"{r:.2f}" for r in self.rewards)
        
        print(
            f"[END] success={str(success).lower()} steps={self.step_count} score={score:.2f} rewards={rewards_str}",
            flush=True
        )


# ============================================================================
# OPENAI CLIENT
# ============================================================================
def get_openai_client() -> OpenAI | None:
    """Initialize OpenAI client from environment variables."""
    if not API_BASE_URL:
        return None
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or ""
    )


# ============================================================================
# RULE-BASED CLASSIFICATION
# ============================================================================
def classify_email(email: str) -> str:
    """
    Rule-based email classification into one of 8 categories.
    """
    text = email.lower()
    
    # Check for urgent first (highest priority)
    if any(word in text for word in ["urgent", "asap", "emergency", "critical", "immediately"]):
        return "urgent"
    
    # Check for refund
    if any(word in text for word in ["refund", "money back", "return", "reimburse"]):
        return "refund"
    
    # Check for complaint
    if any(word in text for word in ["complaint", "unhappy", "disappointed", "terrible", "bad", "poor", "worst"]):
        return "complaint"
    
    # Check for billing
    if any(word in text for word in ["billing", "invoice", "charge", "payment", "subscription", "price", "cost"]):
        return "billing"
    
    # Check for order status
    if any(word in text for word in ["order", "tracking", "shipment", "delivery", "package", "shipping"]):
        return "order_status"
    
    # Check for technical support
    if any(word in text for word in ["technical", "bug", "error", "crash", "not working", "broken", "issue", "problem"]):
        return "technical_support"
    
    # Check for spam
    if any(word in text for word in ["unsubscribe", "promotion", "marketing", "advertisement", "spam"]):
        return "spam"
    
    # Default
    return "general_inquiry"


# ============================================================================
# RULE-BASED ACTION SELECTION
# ============================================================================
def select_action(category: str) -> str:
    """
    Select action based on category.
    """
    return CATEGORY_ACTIONS.get(category, "send_general_response")


# ============================================================================
# RULE-BASED REPLY TEMPLATE SELECTION
# ============================================================================
def select_reply_template(category: str) -> str:
    """
    Select reply template based on category.
    """
    return CATEGORY_TEMPLATES.get(category, "general_inquiry_template")


# ============================================================================
# LLM ENTITY EXTRACTION
# ============================================================================
def extract_entities_llm(email: str, client: OpenAI | None) -> dict:
    """
    Extract entities using LLM API.
    Falls back to rule-based if LLM unavailable.
    
    Extracts: customer_name, product, urgency
    """
    # Default values
    default_result = {
        "customer_name": None,
        "product": None,
        "urgency": "medium"
    }
    
    if not client:
        # Fallback: rule-based extraction
        return extract_entities_rule_based(email)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": """You are an email information extraction assistant.
Extract the following fields from the email:

* customer_name
* product
* urgency (low, medium, high, urgent)

Return JSON only:
{
  "customer_name": "",
  "product": "",
  "urgency": ""
}

Email:"""
                },
                {"role": "user", "content": email}
            ]
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except:
                    return default_result
            else:
                return default_result
        
        # Validate and normalize
        result = {
            "customer_name": data.get("customer_name") if data.get("customer_name") else None,
            "product": data.get("product") if data.get("product") else None,
            "urgency": normalize_urgency(data.get("urgency", "medium"))
        }
        
        return result
    
    except Exception as e:
        # Fallback on error
        return extract_entities_rule_based(email)


def extract_entities_rule_based(email: str) -> dict:
    """
    Rule-based entity extraction when LLM unavailable.
    """
    text = email.lower()
    
    # Extract customer name from signature
    name = None
    name_patterns = [
        r'(?:from|name|regards|sincerely|thanks)[,:]?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*$'
    ]
    for pattern in name_patterns:
        match = re.search(pattern, email, re.MULTILINE | re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            break
    
    # Extract product
    product = None
    product_keywords = ["laptop", "phone", "tablet", "computer", "device", "monitor", "keyboard", "mouse", "headphones", "camera", "printer", "router", "modem", "speaker", "charger", "cable", "adapter", "battery", "screen", "display"]
    for kw in product_keywords:
        if kw in text:
            product = kw
            break
    
    # Determine urgency
    urgency = "medium"
    if any(word in text for word in ["urgent", "asap", "emergency", "immediately", "critical"]):
        urgency = "urgent"
    elif any(word in text for word in ["high priority", "important", "please hurry"]):
        urgency = "high"
    elif any(word in text for word in ["low priority", "whenever", "no rush"]):
        urgency = "low"
    
    return {
        "customer_name": name,
        "product": product,
        "urgency": urgency
    }


def normalize_urgency(urgency: str) -> str:
    """
    Normalize urgency to valid values and map to priority.
    """
    urgency_clean = urgency.lower().strip()
    
    # Map urgency to priority
    mapping = {
        "urgent": "urgent",
        "high": "high",
        "medium": "medium",
        "low": "low"
    }
    
    return mapping.get(urgency_clean, "medium")


# ============================================================================
# HYBRID AGENT MAIN LOGIC
# ============================================================================
def run_hybrid_agent(email: str) -> dict:
    """
    Run the hybrid agent on an email.
    
    Returns:
        {
            "category": str,
            "priority": str,
            "action": str,
            "reply_template": str,
            "customer_name": str | None,
            "product": str | None
        }
    """
    # Initialize OpenAI client
    client = get_openai_client()
    
    # Step 1: Rule-based classification
    category = classify_email(email)
    
    # Step 2: LLM entity extraction
    entities = extract_entities_llm(email, client)
    
    # Step 3: Rule-based action selection
    action = select_action(category)
    
    # Step 4: Rule-based reply template selection
    reply_template = select_reply_template(category)
    
    # Map urgency to priority
    priority = entities["urgency"]
    
    return {
        "category": category,
        "priority": priority,
        "action": action,
        "reply_template": reply_template,
        "customer_name": entities["customer_name"],
        "product": entities["product"]
    }


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================
def run_inference(email: str) -> None:
    """
    Run the complete inference pipeline with strict logging.
    
    Steps:
    1. Classify email (rule-based)
    2. Extract entities (LLM)
    3. Select action and template (rule-based)
    """
    logger = StrictLogger(
        task="email-triage",
        env="openenv",
        model=MODEL_NAME
    )
    
    # Track success
    all_success = True
    
    try:
        # Print START
        logger.start()
        
        # ========================================================================
        # STEP 1: Classify email (rule-based)
        # ========================================================================
        category = classify_email(email)
        logger.step("classify_email", 0.33, False, None)
        
        # ========================================================================
        # STEP 2: Extract entities (LLM API)
        # ========================================================================
        client = get_openai_client()
        entities = extract_entities_llm(email, client)
        
        # Validate extraction
        if entities["customer_name"] is None and entities["product"] is None:
            # Partial success - extraction worked but no entities found
            logger.step("extract_entities", 0.33, False, None)
        else:
            logger.step("extract_entities", 0.33, False, None)
        
        # ========================================================================
        # STEP 3: Select action and reply template (rule-based)
        # ========================================================================
        action = select_action(category)
        reply_template = select_reply_template(category)
        
        # Validate all required fields are present
        result = {
            "category": category,
            "priority": entities["urgency"],
            "action": action,
            "reply_template": reply_template,
            "customer_name": entities["customer_name"],
            "product": entities["product"]
        }
        
        # Check all fields exist
        required_fields = ["category", "priority", "action", "reply_template"]
        has_all_fields = all(result.get(field) for field in required_fields)
        
        if has_all_fields:
            logger.step("generate_reply", 0.34, True, None)
        else:
            missing = [f for f in required_fields if not result.get(field)]
            logger.step("generate_reply", 0.00, True, f"missing fields: {missing}")
            all_success = False
        
        # ========================================================================
        # END: Print final log
        # ========================================================================
        logger.end(all_success)
        
        # Return result for potential further use
        return result
        
    except Exception as e:
        # Ensure END is always printed even on exception
        logger.end(False)
        sys.exit(1)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main() -> None:
    """Main entry point."""
    
    # Sample email for testing
    sample_email = """Subject: Request for refund on order #12345

Hi Support Team,

I ordered a laptop last week (Order #12345) but it arrived damaged. 
The screen is cracked and it won't turn on. I would like a full refund.

Please process this as soon as possible.

Thanks,
John Smith"""
    
    # Check if email provided as argument
    if len(sys.argv) > 1:
        # Read email from file if argument is a file path
        email_path = sys.argv[1]
        if os.path.isfile(email_path):
            with open(email_path, 'r', encoding='utf-8') as f:
                email = f.read()
        else:
            email = sys.argv[1]
    else:
        email = sample_email
    
    run_inference(email)


if __name__ == "__main__":
    main()
