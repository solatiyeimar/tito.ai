from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class BotProcessManager(ABC):
    @abstractmethod
    async def start_bot(
        self, room_url: str, token: str, args: List[str], env_vars: Optional[Dict[str, str]] = None
    ) -> int:
        """Starts a bot process and returns its PID"""
        pass

    @abstractmethod
    def get_status(self, pid: int) -> str:
        """Returns 'running' or 'finished' (or raises error)"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Clean up finished processes"""
        pass
