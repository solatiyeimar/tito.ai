from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from app.dependencies import get_call_service
from app.Domains.Call.Models.call import CallConfig
from app.Domains.Call.Services.call_service import CallService
from app.Http.DTOs.error_schemas import APIErrorResponse
from app.Http.DTOs.schemas import CallRequest, CallResponse, Link

router = APIRouter(tags=["Calls"])


class ConnectRequest(BaseModel):
    variables: Optional[Dict[str, Any]] = None


@router.post(
    "/calls",
    response_model=CallResponse,
    status_code=201,
    summary="Create a new call",
    description="Spawns a new agent process for a specific assistant.",
    responses={404: {"model": APIErrorResponse}, 422: {"model": APIErrorResponse}},
)
async def create_call(
    request: Request, body: CallRequest, service: CallService = Depends(get_call_service)
):
    """
    Initiates a new call by starting a bot process with the specified assistant configuration.
    """
    # Map DTO to Domain Model
    config = CallConfig(
        assistant_id=body.assistant_id,
        phone_number=body.phone_number,
        variables=body.variables,
        dynamic_vocabulary=body.dynamic_vocabulary,
        secrets=body.secrets,
    )

    try:
        session = await service.initiate_call(config)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Call failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    base_url = str(request.base_url).rstrip("/")
    return CallResponse(
        id=session.id,
        status=session.status,
        room_url=session.room_url,
        token=session.token,
        _links=[
            Link(href=f"{base_url}/status/{session.id}", method="GET", rel="status"),
            Link(href=f"{base_url}/assistants/{body.assistant_id}", method="GET", rel="assistant"),
        ],
    )


@router.post(
    "/connect",
    summary="RTVI connection (dynamic)",
    description="API-friendly endpoint returning connection credentials for RTVI clients.",
    responses={422: {"model": APIErrorResponse}},
)
async def rtvi_connect(
    request: Request, service: CallService = Depends(get_call_service)
) -> Dict[str, Any]:
    """
    Dynamic connection endpoint that accepts inline configuration for the bot.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    session = await service.start_rtvi_session(body)

    return {
        "room_url": session.room_url,
        "token": session.token,
        "bot_pid": int(session.id),
        "status_endpoint": f"/status/{session.id}",
    }


@router.post(
    "/connect/{assistant_id}",
    summary="RTVI connection with assistant",
    description="Start a bot using a pre-defined assistant configuration for RTVI.",
    responses={404: {"model": APIErrorResponse}, 422: {"model": APIErrorResponse}},
)
async def connect_assistant(
    assistant_id: str,
    request: Request,
    body: Optional[ConnectRequest] = None,
    service: CallService = Depends(get_call_service),
):
    """
    Connects an RTVI client to a bot process based on an assistant ID.
    """
    variables = body.variables if body else None
    config = CallConfig(assistant_id=assistant_id, variables=variables)

    try:
        session = await service.initiate_call(config)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "room_url": session.room_url,
        "token": session.token,
        "bot_pid": int(session.id),
        "status_endpoint": f"/status/{session.id}",
    }


@router.get(
    "/status/{pid}", summary="Get process status", responses={404: {"model": APIErrorResponse}}
)
def get_status(pid: str, service: CallService = Depends(get_call_service)):
    """
    Check if a bot process is still running or has finished.
    """
    try:
        status = service.get_call_status(pid)
        return JSONResponse({"bot_id": pid, "status": status})
    except Exception:
        raise HTTPException(status_code=404, detail=f"Bot with PID {pid} not found")

    """
    Check if a bot process is still running or has finished.
    """
    proc_tuple = bot_procs.get(pid)
    if not proc_tuple:
        raise HTTPException(status_code=404, detail=f"Bot with PID {pid} not found")
    proc, _ = proc_tuple
    status = "running" if proc.poll() is None else "finished"
    return JSONResponse({"bot_id": pid, "status": status})
