"""
Main entry point for the FastAPI server.

This module defines the FastAPI application, its endpoints,
and lifecycle management. It relies on environment variables for configuration
(e.g., HOST, FAST_API_PORT) and uses run_helpers.py for bot startup.
"""

import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from loguru import logger

from app.Core.Config.server import ServerConfig
from app.dependencies import get_call_service, get_process_manager, start_process_cleanup
from app.Domains.Call.Models.call import CallConfig
from app.Http.Routes.assistants import router as assistants_router
from app.Http.Routes.calls import router as calls_router
from app.Http.Routes.campaigns import router as campaigns_router
from app.Http.Routes.ws.voice import router as voice_ws_router

# Server configuration
server_config = ServerConfig()

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager that handles startup and shutdown tasks.
    It initializes the background cleanup task.
    """
    cleanup_task = asyncio.create_task(start_process_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        # Close room provider session if needed
        # (Assuming DailyRoomProvider manages its own session lifecycle per request or via singleton close if exposed)


from app.Core.Exceptions.handlers import register_exception_handlers

# Create the FastAPI app with the lifespan context
app: FastAPI = FastAPI(
    lifespan=lifespan,
    title="Tito.ai Agent API",
    description="API for managing AI agents, calls, and campaigns.",
    version="1.0.0",
)

# Register custom exception handlers for standardized error responses
register_exception_handlers(app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assistants_router)
app.include_router(calls_router)
app.include_router(campaigns_router)
app.include_router(voice_ws_router)


def parse_server_args():
    """Parse server-specific arguments and store remaining args for bot processes"""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host", help="Server host")
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Parse known server args and keep remaining for bots
    server_args, remaining_args = parser.parse_known_args()

    # Update server config with parsed args
    global server_config
    if server_args.host:
        server_config.host = server_args.host
    if server_args.port:
        server_config.port = server_args.port
    if server_args.reload:
        server_config.reload = server_args.reload

    # Pass remaining args to ProcessManager
    get_process_manager().set_base_args(remaining_args)


# Call this before starting the app
parse_server_args()


@app.get(
    "/",
    summary="Start Agent (Browser)",
    description="Endpoint for direct browser access. Creates a Daily room and redirects the user to it after spawning a bot.",
)
async def start_agent(request: Request):
    """
    Creates a room and spawns a bot subprocess, then redirects to the room URL.
    This is a quick demo endpoint that starts a bot with default config (or whatever args passed).
    """
    logger.info("Creating room for bot (browser access)")

    # We use CallService directly here (Service Locator pattern for this root endpoint simplicity)
    service = get_call_service()

    # Start a generic RTVI session
    try:
        session = await service.start_rtvi_session({})
        logger.info(f"Room URL: {session.room_url}")
        return RedirectResponse(session.room_url)
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server")
    uvicorn.run(
        "main:app",
        host=server_config.host,
        port=server_config.port,
        reload=server_config.reload,
    )
