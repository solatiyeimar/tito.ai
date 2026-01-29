from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.dependencies import get_assistant_service
from app.Domains.Assistant.Services.assistant_service import AssistantService

router = APIRouter(tags=["Voice WebSocket"])


@router.websocket("/assistants/{assistant_id}/voice")
async def websocket_voice_endpoint(
    websocket: WebSocket,
    assistant_id: str,
    service: AssistantService = Depends(get_assistant_service),
):
    """
    WebSocket endpoint for real-time voice interaction.
    """
    await websocket.accept()

    assistant = service.get_assistant(assistant_id)
    if not assistant:
        await websocket.close(code=4004, reason="Assistant not found")
        return

    # TODO: Initialize Pipecat pipeline with WebSocket transport here
    try:
        await websocket.send_text(f"Connected to Voice Agent: {assistant.name}")
        while True:
            # Placeholder loop for audio streaming
            data = await websocket.receive_bytes()
            # Feed to pipeline...
    except WebSocketDisconnect:
        pass
