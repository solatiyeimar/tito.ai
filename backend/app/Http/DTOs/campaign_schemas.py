import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.Http.Responses.hateoas import HateoasModel, Link

# --- DTOs ---


class ContactDTO(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phone: str
    name: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "called", "failed", "completed"] = "pending"
    last_call_id: Optional[str] = None


class CampaignCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    type: Literal["inbound", "outbound"] = "outbound"
    assistant_id: str
    contacts: List[ContactDTO] = Field(default_factory=list)
    concurrency: int = 1


class CampaignResponse(HateoasModel):
    id: str
    name: str
    description: Optional[str] = None
    type: str
    assistant_id: str
    contacts: List[ContactDTO]
    created_at: datetime
    status: str
    concurrency: int

    links: List[Link] = Field(default_factory=list, alias="_links")

    class Config:
        from_attributes = True


# Alias for backward compatibility if needed, or remove if fully refactoring
CampaignConfig = CampaignResponse
