import aiohttp
from fastapi import HTTPException
from pipecat.transports.daily.utils import DailyRESTHelper, DailyRoomParams

from app.Core.Config.server import ServerConfig
from app.Domains.Call.Interfaces.room_provider import RoomProvider


class DailyRoomProvider(RoomProvider):
    def __init__(self):
        self.config = ServerConfig()
        self.helper = None
        self.session = None

    async def _ensure_helper(self):
        if not self.helper:
            self.session = aiohttp.ClientSession()
            self.helper = DailyRESTHelper(
                daily_api_key=self.config.daily_api_key,
                daily_api_url=self.config.daily_api_url,
                aiohttp_session=self.session,
            )

    async def create_room_and_token(self) -> tuple[str, str]:
        await self._ensure_helper()
        try:
            room = await self.helper.create_room(DailyRoomParams())
            if not room.url:
                raise Exception("Daily API returned no URL")

            token = await self.helper.get_token(room.url)
            if not token:
                raise Exception("Daily API returned no token")

            return room.url, token
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create Daily room: {str(e)}")

    async def delete_room(self, room_url: str) -> None:
        await self._ensure_helper()
        try:
            await self.helper.delete_room_by_url(room_url)
        except Exception:
            pass  # Best effort

    async def close(self):
        if self.session:
            await self.session.close()
