import os
import time
import aiohttp

async def create_daily_room() -> tuple[str, str]:
    """Create a Daily room and get token using DAILY_API_KEY from .env."""
    api_key = os.getenv("DAILY_API_KEY")
    api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
    
    if not api_key:
        raise ValueError("DAILY_API_KEY is required in .env to auto-create rooms")
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    async with aiohttp.ClientSession() as session:
        # Create room
        async with session.post(
            f"{api_url}/rooms",
            headers=headers,
            json={"properties": {"exp": int(time.time()) + 3600}}
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to create room: {await resp.text()}")
            room_data = await resp.json()
            room_url = room_data["url"]
            room_name = room_data["name"]
        
        # Get token
        async with session.post(
            f"{api_url}/meeting-tokens",
            headers=headers,
            json={"properties": {"room_name": room_name}}
        ) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to get token: {await resp.text()}")
            token_data = await resp.json()
            token = token_data["token"]
    
    return room_url, token
