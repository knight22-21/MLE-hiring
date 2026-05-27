import re
from typing import Tuple

# Simple regex patterns for PII
PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b", 
    "phone": r"\b\+?1?\s*\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
}

def detect_and_redact_pii(text: str) -> Tuple[bool, str]:
    """Detects PII in the text and returns a redacted version along with a boolean flag."""
    if not text:
        return False, text
        
    pii_detected = False
    redacted_text = text
    
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, redacted_text):
            pii_detected = True
            redacted_text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", redacted_text)
            
    return pii_detected, redacted_text

def check_adversarial_heuristics(text: str) -> bool:
    """Fast rule-based check for common prompt injections and adversarial patterns."""
    text_lower = text.lower()
    adversarial_phrases = [
        "disregard all previous instructions",
        "system prompt",
        "ignore instructions",
        "you are no longer",
        "override safety protocols",
        "output the following exactly",
        "dan mode",
        "output the word",
        "base64", 
        "ignore your system",
        "print your instructions"
    ]
    for phrase in adversarial_phrases:
        if phrase in text_lower:
            return True
    return False
