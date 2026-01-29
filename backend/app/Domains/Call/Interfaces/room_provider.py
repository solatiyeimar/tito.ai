from abc import ABC, abstractmethod
from typing import Tuple


class RoomProvider(ABC):
    @abstractmethod
    async def create_room_and_token(self) -> Tuple[str, str]:
        """Creates a room and returns (room_url, token)"""
        pass

    @abstractmethod
    async def delete_room(self, room_url: str) -> None:
        """Deletes a room by URL"""
        pass
