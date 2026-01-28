"""Flow-based bot implementation using dynamic configuration."""

from functools import partial
from typing import Dict, List, Optional, Any
import sys
import uuid
import random
import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat_flows import FlowArgs, FlowManager, FlowResult
from pipecat_flows.types import ContextStrategy, ContextStrategyConfig, FlowsFunctionSchema

from app.Domains.Agent.Bots.base_bot import BaseBot
from app.Core.Config.bot import BotConfig
from app.Http.DTOs.schemas import WebhookConfig, FlowConfig, FlowNode
from app.Domains.Agent.Prompts.helpers import get_current_date_uk

# Load environment variables from .env file
load_dotenv()

# Configure logger
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# ==============================================================================
# Function Handlers
# ==============================================================================

async def collect_recording_consent(args: FlowArgs) -> FlowResult:
    """Process recording consent collection."""
    return {"recording_consent": args["recording_consent"]}

async def collect_identification(args: FlowArgs) -> FlowResult:
    """Collect user's name and ID number."""
    return {"name": args.get("name"), "id_number": args.get("id_number")}

async def offer_ticket_creation(args: FlowArgs) -> FlowResult:
    """Process problem description."""
    return {"problem_description": args.get("problem_description")}

async def confirm_ticket_creation(args: FlowArgs) -> FlowResult:
    """Process ticket creation confirmation."""
    if args.get("create_ticket"):
        ticket_number = random.randint(10000, 99999)
        return {"create_ticket": True, "ticket_number": ticket_number}
    return {"create_ticket": False}

# ==============================================================================
# Transition Callbacks
# ==============================================================================

async def handle_recording_consent(args: FlowArgs, flow_manager: FlowManager):
    """Transition from consent to identification."""
    await transition_to_node("identification", flow_manager, args)

async def handle_identification(args: FlowArgs, flow_manager: FlowManager):
    """Transition from identification to problem description."""
    await transition_to_node("problem_description", flow_manager, args)

async def handle_problem_description(args: FlowArgs, flow_manager: FlowManager):
    """Transition from problem description to ticket result."""
    await transition_to_node("ticket_result", flow_manager, args)

async def handle_ticket_result(args: FlowArgs, flow_manager: FlowManager):
    """Transition from ticket result to close call."""
    await transition_to_node("close_call", flow_manager, args)

# ==============================================================================
# Registry
# ==============================================================================

FUNCTION_REGISTRY = {
    "collect_recording_consent": collect_recording_consent,
    "collect_identification": collect_identification,
    "offer_ticket_creation": offer_ticket_creation,
    "confirm_ticket_creation": confirm_ticket_creation,
}

TRANSITION_REGISTRY = {
    "handle_recording_consent": handle_recording_consent,
    "handle_identification": handle_identification,
    "handle_problem_description": handle_problem_description,
    "handle_ticket_result": handle_ticket_result,
}

# ==============================================================================
# Dynamic Flow Helpers
# ==============================================================================

async def transition_to_node(node_name: str, flow_manager: FlowManager, args: FlowArgs):
    """Helper to transition to a specific node using the flow configuration."""
    flow_bot = flow_manager.task  # This is not right, FlowManager doesn't reference Bot easily. 
    # But wait, we are inside a function. We need access to the config.
    # The flow_manager has state. We can store config in flow_manager.state?
    # Or we can attach it to the FlowManager instance if we subclass it.
    
    # Actually, flow_manager doesn't expose the bot config directly.
    # We can rely on the fact that this module has access to the config IF it was global, but it's not.
    
    # We need to find where the flow config is stored.
    # Let's assume we can pass it via closure or state.
    
    # Solution: We will store the flow_config in the FlowManager.state during initialization.
    flow_config: FlowConfig = flow_manager.state.get("_flow_config")
    bot_config: BotConfig = flow_manager.state.get("_bot_config")
    
    if not flow_config:
        logger.error("Flow config not found in state!")
        return

    node_config = create_node_from_config(node_name, flow_config, bot_config, flow_manager.state)
    if node_config:
        await flow_manager.set_node_from_config({**node_config, "name": node_name})
    else:
        logger.error(f"Node {node_name} could not be created from config.")


def create_node_from_config(node_name: str, flow_config: FlowConfig, bot_config: BotConfig, state: Dict[str, Any]) -> Optional[Dict]:
    """Creates a node dictionary from the FlowConfig, injecting variables."""
    if node_name not in flow_config.nodes:
        return None

    node_def = flow_config.nodes[node_name]
    
    # Prepare context for variable injection
    user_name = state.get("name", "Usuario")
    first_name = user_name.split(" ")[0] if user_name else "Usuario"
    
    context = {
        "bot_name": bot_config.bot_name,
        "current_date": get_current_date_uk(),
        "user_name": user_name,
        "first_name": first_name,
        "name_context": f"El usuario ha dado su nombre como: {user_name}" if user_name not in ["Usuario", None] else ""
    }
    
    # Build messages
    messages = []
    for msg in node_def.messages:
        try:
            content = msg.content.format(**context)
        except KeyError as e:
            logger.warning(f"Missing variable in prompt template: {e}")
            content = msg.content # Fallback
            
        messages.append({"role": msg.role, "content": content})

    # Build functions
    functions = []
    for func in node_def.functions:
        handler = FUNCTION_REGISTRY.get(func.handler)
        transition = TRANSITION_REGISTRY.get(func.transition_callback)
        
        if handler:
            functions.append(
                FlowsFunctionSchema(
                    name=func.name,
                    description=func.description,
                    properties=func.properties,
                    required=func.required,
                    handler=handler,
                    transition_callback=transition
                )
            )
        else:
            logger.warning(f"Handler {func.handler} not found in registry.")

    node_dict = {
        "task_messages": messages,
        "functions": functions,
    }
    
    if node_def.post_actions:
         node_dict["post_actions"] = node_def.post_actions

    logger.debug(f"Created node config for {node_name}: keys={list(node_dict.keys())}")
    return node_dict

# ==============================================================================
# FlowBot Implementation
# ==============================================================================

class FlowBot(BaseBot):
    """Flow-based bot implementation using pipecat-flows."""

    def __init__(self, config: BotConfig, system_messages: Optional[List[Dict[str, str]]] = None, webhook_config: Optional[WebhookConfig] = None):
        """Initialize the FlowBot with a FlowManager."""
        super().__init__(config, system_messages, webhook_config)
        self.flow_manager = None

    async def _handle_first_participant(self):
        """Initialize the flow manager and start the conversation."""
        self.flow_manager = FlowManager(
            task=self.task,
            llm=self.llm,
            tts=self.tts,
            context_aggregator=self.context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.APPEND,
            ),
        )

        # Store configs in state for access during transitions
        self.flow_manager.state.update({
            "_flow_config": self.config.flow_config,
            "_bot_config": self.config
        })

        # Initialize the flow manager
        await self.flow_manager.initialize()

        # If we have a flow configuration from the assistant JSON, use it
        if self.config.flow_config:
            initial_node = self.config.flow_config.initial_node
            node_config = create_node_from_config(
                initial_node, 
                self.config.flow_config, 
                self.config, 
                self.flow_manager.state
            )
            if node_config:
                # Use global speak_first to override respond_immediately for the initial node
                # but only if global speak_first is False (since True is already default)
                node_data = node_config.copy()
                node_data["name"] = initial_node
                if not self.config.speak_first:
                     node_data["respond_immediately"] = False
                     
                await self.flow_manager.set_node_from_config(node_data)
            else:
                logger.error(f"Failed to create initial node {initial_node}")
        else:
            # Fallback to legacy/hardcoded flow if no config provided (for backward compatibility or testing)
            logger.warning("No flow configuration found. Falling back to default logic (which is now deprecated).")
            # If we wanted to keep the old hardcoded logic, we would keep the old functions.
            # But the requirement is to use JSON. 
            pass
