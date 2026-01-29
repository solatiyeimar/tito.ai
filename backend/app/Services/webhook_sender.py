
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from loguru import logger
from app.Http.DTOs.schemas import WebhookConfig

class WebhookSender:
    def __init__(self, config: Optional[WebhookConfig]):
        self.config = config

    async def send(self, event: str, payload: Dict[str, Any]):
        if not self.config or not self.config.url:
            return

        if event not in self.config.events:
            return

        data = {
            "event": event,
            "data": payload,
            "timestamp": asyncio.get_event_loop().time() # Or use real time
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.url,
                    json=data,
                    headers=self.config.headers
                ) as resp:
                    if resp.status >= 400:
                        logger.warning(f"Webhook {event} failed with status {resp.status}: {await resp.text()}")
                    else:
                        logger.debug(f"Webhook {event} sent successfully")
        except Exception as e:
            logger.error(f"Failed to send webhook {event}: {e}")
