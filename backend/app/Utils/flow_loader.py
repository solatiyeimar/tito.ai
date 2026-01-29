from typing import Any, Dict, List, Optional, Tuple, Callable
from loguru import logger
import functools

try:
    from pipecat_flows import (
        FlowArgs,
        FlowManager,
        FlowResult,
        FlowsFunctionSchema,
        NodeConfig,
        ContextStrategy,
        ContextStrategyConfig,
    )
except ImportError:
    logger.error("pipecat-flows is not installed. Dynamic flows will not work.")

def build_handler(next_node_id: str, flow_data: Dict[str, Any]) -> Callable:
    """Builds a generic handler that transitions to the next node."""
    
    async def handler(args: FlowArgs, flow_manager: FlowManager) -> Tuple[Any, NodeConfig]:
        logger.info(f"Transitioning to node: {next_node_id} with args: {args}")
        # Build the next node config
        next_node = build_node_config(next_node_id, flow_data)
        if not next_node:
            raise ValueError(f"Could not transition to node {next_node_id}: Node not found.")
            
        # Current implementation assumes no complex logic in transition, 
        # but we could return args as result if needed.
        # For simple transitions, we just return None and the next node.
        return None, next_node
        
    return handler

def build_node_config(node_id: str, flow_data: Dict[str, Any]) -> Optional[NodeConfig]:
    """Finds a node by ID in flow_data and builds a NodeConfig using new conventions."""
    
    nodes = flow_data.get("nodes", [])
    node_def = next((n for n in nodes if n["id"] == node_id), None)
    
    if not node_def:
        logger.error(f"Node definition for '{node_id}' not found.")
        return None
    
    data = node_def.get("data", {})
    
    # Extract messages - new format supports both role_messages and task_messages
    role_messages = data.get("role_messages", [])
    task_messages = data.get("task_messages", [])
    
    # Legacy support: convert old "messages" format to new format
    if not task_messages and "messages" in data:
        task_messages = data["messages"]
    
    # Extract functions with new handler format
    functions = []
    for func_def in data.get("functions", []):
        func_name = func_def.get("name")
        next_node_id = func_def.get("next_node_id")
        
        if not func_name:
            continue
            
        # Build the wrapper handler using new conventions
        handler = build_handler(next_node_id, flow_data)
        
        functions.append(
            FlowsFunctionSchema(
                name=func_name,
                handler=handler,
                description=func_def.get("description", ""),
                properties=func_def.get("properties", {}),
                required=func_def.get("required", [])
            )
        )
    
    # Extract context strategy if specified
    context_strategy = None
    if "context_strategy" in data:
        strategy_config = data["context_strategy"]
        if isinstance(strategy_config, str):
            # Simple string format
            strategy = getattr(ContextStrategy, strategy_config.upper(), ContextStrategy.APPEND)
            context_strategy = ContextStrategyConfig(strategy=strategy)
        elif isinstance(strategy_config, dict):
            # Full configuration
            strategy = getattr(ContextStrategy, strategy_config.get("strategy", "APPEND").upper(), ContextStrategy.APPEND)
            context_strategy = ContextStrategyConfig(
                strategy=strategy,
                summary_prompt=strategy_config.get("summary_prompt")
            )
    
    return NodeConfig(
        name=node_id,
        role_messages=role_messages,
        task_messages=task_messages,
        functions=functions,
        pre_actions=data.get("pre_actions", []),
        post_actions=data.get("post_actions", []),
        context_strategy=context_strategy,
        respond_immediately=data.get("respond_immediately", True)
    )

def load_flow_from_json(json_data: Dict[str, Any]) -> NodeConfig:
    """Entry point to load a flow from the pipecat.ai JSON schema."""
    
    # Try to find the initial node
    nodes = json_data.get("nodes", [])
    initial_node = next((n for n in nodes if n.get("type") == "initial"), None)
    
    if not initial_node:
        # Fallback to the first node if no 'initial' type found
        if nodes:
            initial_node = nodes[0]
            logger.warning(f"No node of type 'initial' found. Using the first node: {initial_node.get('id')}")
        else:
            raise ValueError("No nodes found in flow JSON.")
            
    initial_node_id = initial_node.get("id")
    logger.info(f"Loading dynamic flow starting at node: {initial_node_id}")
    
    return build_node_config(initial_node_id, json_data)
