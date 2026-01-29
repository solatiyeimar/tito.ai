from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import json
from loguru import logger
from pydantic import BaseModel

from app.Http.DTOs.schemas import CallRequest, CallResponse, Link
from app.Http.DTOs.error_schemas import APIErrorResponse
from app.Services.assistant_store import AssistantStore
from app.Services.runner_service import start_bot_process, create_room_and_token, bot_procs
from app.Core.Parsers.bot_config_parser import dict_to_cli_args

router = APIRouter(tags=["Calls"])

class ConnectRequest(BaseModel):
    variables: Optional[Dict[str, Any]] = None

@router.post(
    "/calls", 
    response_model=CallResponse, 
    status_code=201,
    summary="Create a new call",
    description="Spawns a new agent process for a specific assistant.",
    responses={404: {"model": APIErrorResponse}, 422: {"model": APIErrorResponse}}
)
async def create_call(request: Request, body: CallRequest):
    """
    Initiates a new call by starting a bot process with the specified assistant configuration.
    """
    logger.info(f"Creating call for assistant: {body.assistant_id}")
    
    store = AssistantStore()
    assistant = store.get_assistant(body.assistant_id)
    
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
        
    config_args = dict_to_cli_args(assistant)
    config_args.extend(["--assistant-id", body.assistant_id])

    if body.variables:
         variables_json = json.dumps(body.variables)
         config_args.extend(["--prompt-variables", variables_json])
         
    if body.dynamic_vocabulary:
        keywords = ",".join(body.dynamic_vocabulary)
        config_args.extend(["--stt-keywords", keywords])
         
    base_url = str(request.base_url).rstrip("/")
    room_url, token = await create_room_and_token()
    pid = await start_bot_process(room_url, token, extra_args=config_args, env_vars=body.secrets)
    
    return CallResponse(
        id=str(pid),
        status="initiated",
        room_url=room_url,
        token=token,
        _links=[
            Link(href=f"{base_url}/status/{pid}", method="GET", rel="status"),
            Link(href=f"{base_url}/assistants/{body.assistant_id}", method="GET", rel="assistant")
        ]
    )

@router.post(
    "/connect",
    summary="RTVI connection (dynamic)",
    description="API-friendly endpoint returning connection credentials for RTVI clients.",
    responses={422: {"model": APIErrorResponse}}
)
async def rtvi_connect(request: Request) -> Dict[str, Any]:
    """
    Dynamic connection endpoint that accepts inline configuration for the bot.
    """
    logger.info("Creating room for RTVI connection")
    
    config_args = []
    try:
        body = await request.json()
        if isinstance(body, dict):
            arg_map = {
                "bot_type": "-a",
                "bot_name": "-n",
                "llm_provider": "-l",
                "llm_model": "-m",
                "llm_temperature": "-T",
                "stt_provider": "-s",
                "tts_provider": "-p",
                "tts_voice": "-v",
            }
            for key, flag in arg_map.items():
                if key in body:
                    config_args.extend([flag, str(body[key])])
            
            if "enable_stt_mute_filter" in body:
                val = "true" if body["enable_stt_mute_filter"] else "false"
                config_args.extend(["--enable-stt-mute-filter", val])
    except Exception:
        pass

    room_url, token = await create_room_and_token()
    pid = await start_bot_process(room_url, token, extra_args=config_args)
    return {
        "room_url": room_url,
        "token": token,
        "bot_pid": pid,
        "status_endpoint": f"/status/{pid}",
    }

@router.post(
    "/connect/{assistant_id}",
    summary="RTVI connection with assistant",
    description="Start a bot using a pre-defined assistant configuration for RTVI.",
    responses={404: {"model": APIErrorResponse}, 422: {"model": APIErrorResponse}}
)
async def connect_assistant(assistant_id: str, request: Request, body: Optional[ConnectRequest] = None):
    """
    Connects an RTVI client to a bot process based on an assistant ID.
    """
    logger.info(f"Connecting to assistant: {assistant_id}")
    
    store = AssistantStore()
    assistant = store.get_assistant(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
        
    config_args = dict_to_cli_args(assistant)
    config_args.extend(["--assistant-id", assistant_id])

    if body and body.variables:
         variables_json = json.dumps(body.variables)
         config_args.extend(["--prompt-variables", variables_json])
        
    room_url, token = await create_room_and_token()
    pid = await start_bot_process(room_url, token, extra_args=config_args)
    
    return {
        "room_url": room_url,
        "token": token,
        "bot_pid": pid,
        "status_endpoint": f"/status/{pid}",
    }

@router.get(
    "/status/{pid}",
    summary="Get process status",
    responses={404: {"model": APIErrorResponse}}
)
def get_status(pid: int):
    """
    Check if a bot process is still running or has finished.
    """
    proc_tuple = bot_procs.get(pid)
    if not proc_tuple:
        raise HTTPException(status_code=404, detail=f"Bot with PID {pid} not found")
    proc, _ = proc_tuple
    status = "running" if proc.poll() is None else "finished"
    return JSONResponse({"bot_id": pid, "status": status})

