import os
import sys
import asyncio
import subprocess
from typing import Dict, Any, Tuple
from loguru import logger
from fastapi import HTTPException
from pipecat.transports.daily.utils import DailyRESTHelper, DailyRoomParams
from app.Core.Config.server import ServerConfig

# Runtime state
bot_procs: Dict[int, Tuple[subprocess.Popen, str]] = {}  # {pid: (process, room_url)}
daily_helpers: Dict[str, DailyRESTHelper] = {}  # {key: helper}
server_config = ServerConfig()
bot_args: list[str] = []

async def create_room_and_token() -> Tuple[str, str]:
    """
    Create a Daily room and get an access token.
    """
    if "rest" not in daily_helpers:
         raise HTTPException(status_code=500, detail="DailyRESTHelper not initialized")
         
    helper = daily_helpers["rest"]
    room = await helper.create_room(DailyRoomParams())
    if not room.url:
        raise HTTPException(status_code=500, detail="Failed to create room")
    token = await helper.get_token(room.url)
    if not token:
        raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room.url}")
    return room.url, token

async def start_bot_process(room_url: str, token: str, extra_args: list[str] = None, env_vars: Dict[str, str] = None) -> int:
    """Start a bot subprocess with forwarded CLI arguments"""
    # Check room capacity
    num_bots_in_room = sum(
        1 for proc, url in bot_procs.values() if url == room_url and proc.poll() is None
    )
    if num_bots_in_room >= server_config.max_bots_per_room:
        raise HTTPException(
            status_code=429,
            detail=f"Room {room_url} at capacity ({server_config.max_bots_per_room} bots)",
        )

    try:
        # This file is in backend/app/Services/runner_service.py
        # bot_runner.py is in backend/runners/bot_runner.py
        server_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        run_helpers_path = os.path.join(server_dir, "runners", "bot_runner.py")

        # Build command with forwarded arguments
        cmd = [
            sys.executable,
            run_helpers_path,
            "-u",
            room_url,
            "-t",
            token,
            *bot_args,  # Forward stored CLI arguments from server startup
        ]
        
        if extra_args:
            cmd.extend(extra_args)

        # Set up environment with proper Python path
        env = os.environ.copy()
        env["PYTHONPATH"] = server_dir  # Set server directory as Python path
        
        # Inject provided environment variables (secrets)
        if env_vars:
            env.update(env_vars)

        proc = subprocess.Popen(
            cmd,
            bufsize=1,
            cwd=server_dir,  # Run from server directory
            env=env,
        )
        bot_procs[proc.pid] = (proc, room_url)
        return proc.pid
    except Exception as e:
        logger.error(f"Bot startup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start bot process: {e}")

async def cleanup_finished_processes() -> None:
    """
    Background task to clean up finished bot processes and delete their Daily rooms.
    """
    while True:
        try:
            for pid in list(bot_procs.keys()):
                proc, room_url = bot_procs[pid]
                if proc.poll() is not None:
                    logger.info(f"Cleaning up finished bot process {pid} for room {room_url}")
                    try:
                        if "rest" in daily_helpers:
                             await daily_helpers["rest"].delete_room_by_url(room_url)
                             logger.success(f"Successfully deleted room {room_url}")
                    except Exception as e:
                        logger.error(f"Failed to delete room {room_url}: {str(e)}")
                    del bot_procs[pid]
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        await asyncio.sleep(5)
