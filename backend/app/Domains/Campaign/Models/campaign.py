import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# Value Object
class Contact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phone: str
    name: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "called", "failed", "completed"] = "pending"
    last_call_id: Optional[str] = None


# Aggregate Root
class Campaign(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    type: Literal["inbound", "outbound"] = "outbound"
    assistant_id: str
    contacts: List[Contact] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["draft", "active", "paused", "completed"] = "draft"
    concurrency: int = 1
