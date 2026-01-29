from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CallConfig(BaseModel):
    assistant_id: str
    phone_number: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None
    dynamic_vocabulary: Optional[List[str]] = None
    secrets: Optional[Dict[str, str]] = None


class CallSession(BaseModel):
    id: str  # PID or UUID
    room_url: str
    token: str
    status: str
