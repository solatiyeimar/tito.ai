import asyncio
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException
from loguru import logger

from app.Core.Config.server import ServerConfig
from app.Domains.Call.Interfaces.bot_process_manager import BotProcessManager
from app.Domains.Call.Interfaces.room_provider import RoomProvider


class LocalBotProcessManager(BotProcessManager):
    def __init__(self, room_provider: RoomProvider):
        self.active_processes: Dict[int, Tuple[subprocess.Popen, str]] = (
            {}
        )  # PID -> (Proc, RoomURL)
        self.config = ServerConfig()
        self.room_provider = room_provider
        # Store initial args passed to server to propagate them if needed
        self.base_bot_args = []

    def set_base_args(self, args: List[str]):
        self.base_bot_args = args

    async def start_bot(
        self, room_url: str, token: str, args: List[str], env_vars: Optional[Dict[str, str]] = None
    ) -> int:
        # Capacity check
        active_in_room = sum(
            1
            for proc, url in self.active_processes.values()
            if url == room_url and proc.poll() is None
        )
        if active_in_room >= self.config.max_bots_per_room:
            raise HTTPException(status_code=429, detail="Room capacity reached")

        try:
            # Locate bot runner script
            # Assuming current file is in app/Infrastructure/Call/
            # and bot runner is in backend/runners/bot_runner.py
            backend_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
            runner_path = os.path.join(backend_root, "runners", "bot_runner.py")

            cmd = [sys.executable, runner_path, "-u", room_url, "-t", token, *self.base_bot_args]
            if args:
                cmd.extend(args)

            env = os.environ.copy()
            env["PYTHONPATH"] = backend_root
            if env_vars:
                env.update(env_vars)

            proc = subprocess.Popen(cmd, bufsize=1, cwd=backend_root, env=env)

            self.active_processes[proc.pid] = (proc, room_url)
            return proc.pid

        except Exception as e:
            logger.error(f"Failed to spawn bot: {e}")
            raise HTTPException(status_code=500, detail=f"Bot spawn failed: {e}")

    def get_status(self, pid: int) -> str:
        if pid not in self.active_processes:
            raise HTTPException(status_code=404, detail="Bot process not found")

        proc, _ = self.active_processes[pid]
        return "running" if proc.poll() is None else "finished"

    async def cleanup(self):
        """Periodic cleanup task"""
        while True:
            try:
                for pid in list(self.active_processes.keys()):
                    proc, room_url = self.active_processes[pid]
                    if proc.poll() is not None:
                        logger.info(f"🧹 Cleaning up process {pid} for room {room_url}")
                        await self.room_provider.delete_room(room_url)
                        del self.active_processes[pid]
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
            await asyncio.sleep(5)
