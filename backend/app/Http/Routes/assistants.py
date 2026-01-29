from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.dependencies import get_assistant_service
from app.Domains.Assistant.Models.assistant import Assistant
from app.Domains.Assistant.Services.assistant_service import AssistantService
from app.Http.Responses.hateoas import HateoasModel, Link

router = APIRouter(tags=["Assistants"])

# --- DTOs ---


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class AssistantResponseDTO(HateoasModel):
    id: str
    name: str
    description: Optional[str] = None
    agent: dict
    io_layer: dict
    pipeline_settings: dict
    created_at: str

    # Re-declare to force order at the end
    links: List[Link] = Field(default_factory=list, alias="_links")

    class Config:
        from_attributes = True


def _map_to_response(assistant: Assistant, request: Request) -> AssistantResponseDTO:
    # Manual mapping to ensure safety
    data = assistant.model_dump()
    data["created_at"] = str(data["created_at"])

    dto = AssistantResponseDTO(**data)
    base = str(request.base_url).rstrip("/")
    dto.add_link("self", f"{base}/assistants/{assistant.id}", "GET")
    dto.add_link("chat", f"{base}/assistants/{assistant.id}/chat", "POST")
    dto.add_link(
        "voice_ws", f"{base.replace('http', 'ws')}/assistants/{assistant.id}/voice", "WEBSOCKET"
    )
    return dto


# --- Endpoints ---


@router.get("/assistants", response_model=List[AssistantResponseDTO])
async def list_assistants(
    request: Request, service: AssistantService = Depends(get_assistant_service)
):
    assistants = service.list_assistants()
    return [_map_to_response(a, request) for a in assistants]


@router.post("/assistants", response_model=AssistantResponseDTO, status_code=201)
async def create_assistant(
    assistant: Assistant,
    request: Request,
    service: AssistantService = Depends(get_assistant_service),
):
    created = service.create_assistant(assistant)
    return _map_to_response(created, request)


@router.get("/assistants/{assistant_id}", response_model=AssistantResponseDTO)
async def get_assistant(
    assistant_id: str, request: Request, service: AssistantService = Depends(get_assistant_service)
):
    assistant = service.get_assistant(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return _map_to_response(assistant, request)


@router.put("/assistants/{assistant_id}", response_model=AssistantResponseDTO)
async def update_assistant(
    assistant_id: str,
    update_data: dict,
    request: Request,
    service: AssistantService = Depends(get_assistant_service),
):
    updated = service.update_assistant(assistant_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return _map_to_response(updated, request)


@router.delete("/assistants/{assistant_id}")
async def delete_assistant(
    assistant_id: str, service: AssistantService = Depends(get_assistant_service)
):
    success = service.delete_assistant(assistant_id)
    if not success:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return JSONResponse({"message": "Deleted successfully"})


@router.post("/assistants/{assistant_id}/chat", response_model=ChatResponse)
async def chat_with_assistant(
    assistant_id: str, body: ChatRequest, service: AssistantService = Depends(get_assistant_service)
):
    try:
        response_text = await service.chat_with_assistant(assistant_id, body.message)
        return ChatResponse(response=response_text)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
