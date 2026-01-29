from fastapi import APIRouter, HTTPException, Request
from typing import List
from app.Services.assistant_store import AssistantStore
from app.Http.DTOs.schemas import AssistantConfig, AssistantResponse, Link
from app.Http.DTOs.error_schemas import APIErrorResponse

router = APIRouter(prefix="/assistants", tags=["Assistants"])
store = AssistantStore()

def add_links(assistant: AssistantConfig, request: Request) -> AssistantResponse:
    base_url = str(request.base_url).rstrip("/")
    links = [
        Link(href=f"{base_url}/assistants/{assistant.id}", method="GET", rel="self"),
        Link(href=f"{base_url}/assistants/{assistant.id}", method="PUT", rel="update"),
        Link(href=f"{base_url}/assistants/{assistant.id}", method="DELETE", rel="delete"),
        Link(href=f"{base_url}/calls", method="POST", rel="call"),
    ]
    # Convert to dict, add _links, then validate as AssistantResponse
    data = assistant.model_dump()
    data["_links"] = links
    return AssistantResponse(**data)

@router.get(
    "", 
    response_model=List[AssistantResponse],
    summary="List all assistants",
    description="Retrieve a list of all configured AI assistants with their HATEOAS links."
)
async def list_assistants(request: Request):
    """
    Returns a list of all available assistants in the store.
    """
    assistants = store.list_assistants()
    return [add_links(a, request) for a in assistants]

@router.post(
    "", 
    response_model=AssistantResponse, 
    status_code=201,
    summary="Create a new assistant",
    responses={422: {"model": APIErrorResponse}}
)
async def create_assistant(assistant: AssistantConfig, request: Request):
    """
    Creates a new assistant configuration. 
    Validation errors will return a detail breakdown.
    """
    created = store.create_assistant(assistant)
    return add_links(created, request)

@router.get(
    "/{assistant_id}", 
    response_model=AssistantResponse,
    summary="Get assistant details",
    responses={404: {"model": APIErrorResponse}}
)
async def get_assistant(assistant_id: str, request: Request):
    """
    Retrieve specific assistant details by ID.
    Returns 404 if the assistant does not exist.
    """
    assistant = store.get_assistant(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return add_links(assistant, request)

@router.delete(
    "/{assistant_id}",
    summary="Delete an assistant",
    responses={404: {"model": APIErrorResponse}}
)
async def delete_assistant(assistant_id: str):
    """
    Deletes an assistant configuration by ID.
    """
    store.delete_assistant(assistant_id)
    return {"status": "success", "message": f"Assistant {assistant_id} deleted"}
