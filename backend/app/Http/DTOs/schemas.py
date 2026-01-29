from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Import new Domain Models
from app.Domains.Assistant.Models.assistant import Assistant as AssistantConfig
from app.Domains.Assistant.Models.assistant import WebhookConfig

# --- API Resources (HATEOAS) ---


class Link(BaseModel):
    href: str
    method: str
    rel: str


class CallRequest(BaseModel):
    assistant_id: str
    phone_number: Optional[str] = None  # For SIP/Tel
    variables: Optional[Dict[str, Any]] = None
    secrets: Optional[Dict[str, str]] = None
    dynamic_vocabulary: Optional[List[str]] = None


class CallResponse(BaseModel):
    id: str
    status: str
    room_url: Optional[str] = None
    token: Optional[str] = None
    links: List[Link] = Field(default_factory=list, alias="_links")
