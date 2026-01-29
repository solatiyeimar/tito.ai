"""Simple bot implementation using the base bot framework."""

from typing import Dict, List, Optional

from loguru import logger
from pipecat.frames.frames import LLMRunFrame

from app.Core.Config.bot import BotConfig
from app.Domains.Agent.Bots.base_bot import BaseBot
from app.Domains.Agent.Prompts.simple import get_simple_prompt
from app.Http.DTOs.schemas import WebhookConfig


class SimpleBot(BaseBot):
    """Simple bot implementation with single LLM prompt chain."""

    def __init__(
        self,
        config: BotConfig,
        system_messages: Optional[List[Dict[str, str]]] = None,
        webhook_config: Optional[WebhookConfig] = None,
    ):
        # Define the initial system message if not provided
        if not system_messages:
            system_messages = get_simple_prompt()["task_messages"]

        logger.info(f"Initialising SimpleBot with system messages: {system_messages}")
        super().__init__(config, system_messages, webhook_config)

    async def _handle_first_participant(self):
        """Handle actions when the first participant joins."""
        import asyncio

        from pipecat.frames.frames import LLMMessagesUpdateFrame

        # Apply initial delay if configured
        if self.config.initial_delay > 0:
            logger.info(f"Waiting {self.config.initial_delay}s before greeting...")
            await asyncio.sleep(self.config.initial_delay)

        # Queue the context frame
        frames = [self.context_aggregator.user()._get_context_frame()]

        # Trigger the first response only if speak_first is enabled
        if self.config.speak_first:
            if self.config.initial_message:
                logger.info(f"Speaking initial message: {self.config.initial_message}")
                # Use a system prompt to ensure the LLM says exactly what we want
                # and keeps it in the conversation history
                messages = list(self.context.messages)
                messages.append(
                    {
                        "role": "system",
                        "content": f"The session has started. Say exactly this initial greeting: '{self.config.initial_message}'",
                    }
                )
                frames.append(LLMMessagesUpdateFrame(messages=messages))
            else:
                frames.append(LLMRunFrame())

        # If not interruptible, we could temporarily change task params if task was available
        # However, PipelineTask params are usually set at creation.
        # For now, we focus on the message and delay.

        await self.task.queue_frames(frames)
