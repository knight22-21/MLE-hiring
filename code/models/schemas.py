from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

class ToolCall(BaseModel):
    action: str
    parameters: Dict[str, Any]

class SupportTicketOutput(BaseModel):
    status: Literal["replied", "escalated"]
    product_area: str = Field(..., description="The most relevant support category or domain area")
    response: str = Field(..., description="The user-facing answer. Empty if escalated and no message needed.")
    justification: str = Field(..., description="Reasoning for the decision")
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    source_documents: str = Field(..., description="Pipe-separated file paths of corpus documents used, or empty string")
    risk_level: Literal["low", "medium", "high", "critical"]
    pii_detected: bool = Field(..., description="True if PII was detected, False otherwise")
    language: str = Field(..., description="ISO 639-1 language code")
    actions_taken: List[ToolCall] = Field(default_factory=list, description="List of API tool calls")

    def to_csv_dict(self, issue: str, subject: str, company: str) -> dict:
        import json
        return {
            "issue": issue,
            "subject": subject,
            "company": company,
            "response": self.response,
            "product_area": self.product_area,
            "status": self.status,
            "request_type": self.request_type,
            "justification": self.justification,
            "confidence_score": f"{self.confidence_score:.2f}",
            "source_documents": self.source_documents,
            "risk_level": self.risk_level,
            "pii_detected": "true" if self.pii_detected else "false",
            "language": self.language,
            "actions_taken": json.dumps([a.model_dump() for a in self.actions_taken])
        }
