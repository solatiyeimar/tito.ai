from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import uuid

class Contact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phone: str
    name: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "called", "failed", "completed"] = "pending"
    last_call_id: Optional[str] = None

class CampaignConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    type: Literal["inbound", "outbound"] = "outbound"
    assistant_id: str
    contacts: List[Contact] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["draft", "active", "paused", "completed"] = "draft"
    concurrency: int = 1 # How many calls at a time
    
class CampaignResponse(CampaignConfig):
    pass
