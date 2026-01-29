"""Flow-based bot implementation using dynamic configuration."""

from typing import Dict, List, Optional, Any
import sys
from loguru import logger

from pipecat_flows import FlowManager
from pipecat_flows.types import ContextStrategy, ContextStrategyConfig

from app.Domains.Agent.Bots.base_bot import BaseBot
from app.Core.Config.bot import BotConfig
from app.Http.DTOs.schemas import WebhookConfig

# Configure logger
# logger.remove(0)
# logger.add(sys.stderr, level="DEBUG")

class FlowBot(BaseBot):
    """Flow-based bot implementation using pipecat-flows."""

    def __init__(self, config: BotConfig, system_messages: Optional[List[Dict[str, str]]] = None, webhook_config: Optional[WebhookConfig] = None):
        """Initialize the FlowBot with a FlowManager."""
        super().__init__(config, system_messages, webhook_config)
        self.flow_manager = None

    async def _handle_first_participant(self):
        """Initialize the flow manager and start the conversation using new conventions."""
        from pipecat_flows import ContextStrategy, ContextStrategyConfig
        
        self.flow_manager = FlowManager(
            task=self.task,
            llm=self.llm,
            tts=self.tts,
            context_aggregator=self.context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.APPEND,
            ),
        )

        # Store configs in state for access during transitions if needed by custom handlers
        self.flow_manager.state.update({
            "_bot_config": self.config
        })

        # If we have a flow configuration from the assistant JSON, use it
        if self.config.flow_config:
            flow_data = self.config.flow_config
            
            # If it's a Pydantic model (not expected anymore but for safety), convert to dict
            if hasattr(flow_data, "model_dump"):
                flow_data = flow_data.model_dump()
            
            if isinstance(flow_data, dict) and "nodes" in flow_data:
                from app.Utils.flow_loader import load_flow_from_json
                logger.info("Loading dynamic Pipecat Flow with new conventions.")
                initial_node = load_flow_from_json(flow_data)
                
                # Apply speak_first logic if needed - new convention uses respond_immediately
                if not self.config.speak_first and 'respond_immediately' in initial_node:
                    initial_node['respond_immediately'] = False
                
                await self.flow_manager.initialize(initial_node)
            else:
                logger.error("Invalid flow configuration format. Expected pipecat.ai flow JSON.")
                await self.flow_manager.initialize()
        else:
            logger.warning("No flow configuration provided.")
            await self.flow_manager.initialize()
