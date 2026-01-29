
from typing import List
from app.Http.DTOs.schemas import AssistantConfig

def dict_to_cli_args(config: AssistantConfig) -> List[str]:
    """Convert AssistantConfig to CLI arguments for bot_runner.py"""
    args = []
    
    # Pass architecture type from config
    args.extend(["--architecture-type", config.architecture_type]) 

    if config.name:
        args.extend(["--bot-name", config.name])
        
    if config.agent.provider:
        args.extend(["--llm-provider", config.agent.provider])
    if config.agent.model:
        args.extend(["--llm-model", config.agent.model])
    if config.agent.temperature is not None:
        args.extend(["--llm-temperature", str(config.agent.temperature)])

    if config.io_layer.tts and config.io_layer.tts.provider:
        args.extend(["--tts-provider", config.io_layer.tts.provider])
    if config.io_layer.tts and config.io_layer.tts.voice_id:
        args.extend(["--tts-voice", config.io_layer.tts.voice_id])
        
    if config.io_layer.stt and config.io_layer.stt.provider:
        args.extend(["--stt-provider", config.io_layer.stt.provider])
        
    if config.io_layer.stt and config.io_layer.stt.enable_mute_filter:
        args.extend(["--enable-stt-mute-filter", "true"])
        
    if config.io_layer.sip.amd_enabled:
        args.extend(["--amd-enabled", "true"])
    
    if config.agent.system_prompt:
        args.extend(["--system-prompt", config.agent.system_prompt])
    
    return args
