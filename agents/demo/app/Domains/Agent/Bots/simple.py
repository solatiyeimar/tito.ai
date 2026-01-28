"""Simple bot implementation using the base bot framework."""

from typing import List, Dict, Optional
from pipecat.frames.frames import LLMRunFrame
from app.Domains.Agent.Bots.base_bot import BaseBot
from app.Core.Config.bot import BotConfig
from app.Http.DTOs.schemas import WebhookConfig
from app.Domains.Agent.Prompts.simple import get_simple_prompt
from loguru import logger


class SimpleBot(BaseBot):
    """Simple bot implementation with single LLM prompt chain."""

    def __init__(self, config: BotConfig, system_messages: Optional[List[Dict[str, str]]] = None, webhook_config: Optional[WebhookConfig] = None):
        # Define the initial system message if not provided
        if not system_messages:
            system_messages = get_simple_prompt()["task_messages"]
        
        logger.info(f"Initialising SimpleBot with system messages: {system_messages}")
        super().__init__(config, system_messages, webhook_config)

    async def _handle_first_participant(self):
        """Handle actions when the first participant joins."""
        # Queue the context frame
        frames = [self.context_aggregator.user().get_context_frame()]
        
        # Trigger the first response only if speak_first is enabled
        if self.config.speak_first:
            frames.append(LLMRunFrame())
            
        await self.task.queue_frames(frames)
