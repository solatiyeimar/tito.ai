import argparse
import asyncio
import os
import time
import aiohttp
import json
from typing import Type, Optional
from loguru import logger


from dotenv import load_dotenv
from app.Core.Config.bot import BotConfig

# Load environment variables
load_dotenv()


from app.Utils.daily import create_daily_room



async def run_bot(bot_class: Type, config: BotConfig, room_url: str, token: str, system_messages: Optional[list] = None, webhook_config: Optional['WebhookConfig'] = None) -> None:
    """Universal bot runner handling bot lifecycle.

    Args:
        bot_class: The bot class to instantiate (e.g. FlowBot or SimpleBot)
        config: The configuration instance to use (with bot_type possibly overridden)
        room_url: The Daily room URL
        token: The Daily room token
        system_messages: Optional system messages to initialize the bot with
        webhook_config: Optional webhook configuration
    """
    # Instantiate the bot using the provided configuration instance.
    bot = bot_class(config, system_messages=system_messages, webhook_config=webhook_config)

    # Set up transport and pipeline.
    await bot.setup_transport(room_url, token)
    bot.create_pipeline()

    # Start the bot.
    await bot.start()


def cli() -> None:
    """Parse command-line arguments, override configuration if needed, and start the bot."""
    parser = argparse.ArgumentParser(description="Unified Bot Runner")

    # Optional - if not provided, will auto-create a Daily room
    parser.add_argument("-u", "--room-url", type=str, help="Daily room URL (auto-created if not provided)")
    parser.add_argument("-t", "--token", type=str, help="Authentication token (auto-created if not provided)")

    # Architecture type selection
    parser.add_argument(
        "-a",
        "--architecture-type",
        type=str.lower,
        choices=["simple", "flow", "multimodal"],
        help="Internal architecture of bot (overrides ARCHITECTURE_TYPE)",
    )

    parser.add_argument("-n", "--bot-name", type=str, help="Bot name appearance")

    # LLM configuration
    parser.add_argument(
        "-l",
        "--llm-provider",
        type=str.lower,
        choices=["google", "openai", "anthropic", "groq", "together", "mistral", "aws", "ultravox"],
    )

    parser.add_argument(
        "-m",
        "--llm-model",
        type=str,
        help="Override LLM_MODEL",
    )

    parser.add_argument(
        "-T",
        "--llm-temperature",
        type=float,
        help="Override LLM_TEMPERATURE",
    )

    # STT configuration
    parser.add_argument(
        "-s",
        "--stt-provider",
        type=str.lower,
        choices=["deepgram", "gladia", "assemblyai", "groq"],
        help="Override STT_PROVIDER",
    )

    # TTS configuration
    parser.add_argument(
        "-p",
        "--tts-provider",
        type=str.lower,
        choices=["deepgram", "cartesia", "elevenlabs", "rime", "playht", "openai", "azure"],
        help="Override TTS_PROVIDER",
    )

    # Voice configuration
    parser.add_argument(
        "-v",
        "--tts-voice",
        type=str,
        help="Override TTS voice/id",
    )

    # Backward compatibility args
    parser.add_argument("--google-model", type=str)
    parser.add_argument("--google-temperature", type=float)
    parser.add_argument("--openai-model", type=str)
    parser.add_argument("--openai-temperature", type=float)
    parser.add_argument("--deepgram-voice", type=str)
    parser.add_argument("--cartesia-voice", type=str)
    parser.add_argument("--elevenlabs-voice-id", type=str)
    parser.add_argument("--rime-voice-id", type=str)

    parser.add_argument("--amd-enabled", type=str, help="Enable Answering Machine Detection")
    parser.add_argument("--stt-keywords", type=str, help="Comma-separated list of STT keywords")

    # STT mute filter configuration
    parser.add_argument(
        "--enable-stt-mute-filter",
        type=lambda x: str(x).lower() in ("true", "1", "t", "yes", "y", "on", "enable", "enabled"),
        help="Override ENABLE_STT_MUTE_FILTER (true/false)",
    )

    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Override system prompt",
    )

    parser.add_argument(
        "--assistant-id",
        type=str,
        help="Load configuration from a persisted assistant JSON",
    )

    parser.add_argument(
        "--speak-first",
        type=lambda x: str(x).lower() in ("true", "1", "t", "yes", "y", "on", "enable", "enabled"),
        help="Whether the bot should speak first (default: true)",
    )

    parser.add_argument(
        "--prompt-variables",
        type=str,
        help="JSON string of variables to inject into the system prompt",
    )

    args = parser.parse_args()

    # Helpers to set env vars if config loaded
    def apply_assistant_config(assistant_id: str):
        from app.Services.assistant_store import AssistantStore
        store = AssistantStore()
        assistant = store.get_assistant(assistant_id)
        if not assistant:
            print(f"❌ Assistant {assistant_id} not found!")
            return None
        
        print(f"🤖 Loading assistant: {assistant.name} ({assistant.id})")
        
        # Core
        os.environ["BOT_NAME"] = assistant.name
        os.environ["ARCHITECTURE_TYPE"] = assistant.architecture_type
        
        # Agent / LLM
        os.environ["LLM_PROVIDER"] = assistant.agent.provider
        if assistant.agent.model:
            os.environ["LLM_MODEL"] = assistant.agent.model
        os.environ["LLM_TEMPERATURE"] = str(assistant.agent.temperature)
        
        # IO Layer
        if assistant.io_layer.stt:
            os.environ["STT_PROVIDER"] = assistant.io_layer.stt.provider
        elif assistant.architecture_type == "multimodal":
            os.environ.pop("STT_PROVIDER", None)
            
        if assistant.io_layer.stt and assistant.io_layer.stt.model:
            os.environ["STT_MODEL"] = assistant.io_layer.stt.model 
        if assistant.io_layer.stt and assistant.io_layer.stt.language:
            os.environ["STT_LANGUAGE"] = assistant.io_layer.stt.language
            
        if assistant.io_layer.tts:
            os.environ["TTS_PROVIDER"] = assistant.io_layer.tts.provider
        elif assistant.architecture_type == "multimodal":
            os.environ.pop("TTS_PROVIDER", None)
            
        if assistant.io_layer.tts and assistant.io_layer.tts.voice_id:
            os.environ["TTS_VOICE"] = assistant.io_layer.tts.voice_id
        if assistant.io_layer.tts and assistant.io_layer.tts.language:
            os.environ["TTS_LANGUAGE"] = assistant.io_layer.tts.language

        # Mute filter
        if assistant.io_layer.stt and assistant.io_layer.stt.enable_mute_filter:
            os.environ["ENABLE_STT_MUTE_FILTER"] = "true"

        # Speak first
        os.environ["SPEAK_FIRST"] = "true" if assistant.pipeline_settings.speak_first else "false"

        return assistant

    # If assistant ID provided, load it first to populate defaults in ENV
    loaded_assistant = None
    if args.assistant_id:
        loaded_assistant = apply_assistant_config(args.assistant_id)

    # Set environment variables based on CLI arguments (overrides loaded config)
    if args.architecture_type:
        os.environ["ARCHITECTURE_TYPE"] = args.architecture_type
    if args.bot_name:
        os.environ["BOT_NAME"] = args.bot_name
    
    if args.llm_provider:
        os.environ["LLM_PROVIDER"] = args.llm_provider.lower()
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model
    if args.llm_temperature is not None:
        os.environ["LLM_TEMPERATURE"] = str(args.llm_temperature)
    
    # Map old args to new environment variables for compatibility
    if args.google_model: os.environ["LLM_MODEL"] = args.google_model
    if args.openai_model: os.environ["LLM_MODEL"] = args.openai_model
    if args.openai_temperature: os.environ["LLM_TEMPERATURE"] = str(args.openai_temperature)
    
    if args.stt_provider:
        os.environ["STT_PROVIDER"] = args.stt_provider.lower()
    
    if args.tts_provider:
        os.environ["TTS_PROVIDER"] = args.tts_provider.lower()
    if args.tts_voice:
        os.environ["TTS_VOICE"] = args.tts_voice
    
    # Map old voice args
    if hasattr(args, 'deepgram_voice') and args.deepgram_voice: os.environ["DEEPGRAM_VOICE"] = args.deepgram_voice
    if hasattr(args, 'cartesia_voice') and args.cartesia_voice: os.environ["CARTESIA_VOICE"] = args.cartesia_voice
    if hasattr(args, 'elevenlabs_voice_id') and args.elevenlabs_voice_id: os.environ["ELEVENLABS_VOICE_ID"] = args.elevenlabs_voice_id
    if hasattr(args, 'rime_voice_id') and args.rime_voice_id: os.environ["RIME_VOICE_ID"] = args.rime_voice_id

    if args.enable_stt_mute_filter is not None:
        os.environ["ENABLE_STT_MUTE_FILTER"] = str(args.enable_stt_mute_filter).lower()
        
    if args.amd_enabled:
        os.environ["AMD_ENABLED"] = args.amd_enabled
        
    if args.stt_keywords:
        os.environ["STT_KEYWORDS"] = args.stt_keywords
    
    if args.speak_first is not None:
        os.environ["SPEAK_FIRST"] = "true" if args.speak_first else "false"
        
    # Instantiate the configuration AFTER setting environment variables
    config = BotConfig()

    # If an assistant was loaded, ensure we pass tools to the config object
    if loaded_assistant:
        config.tools = loaded_assistant.agent.tools
        config.flow_config = loaded_assistant.flow

    # Determine the bot class to use based on the configuration
    if config.architecture_type == "flow":
        from app.Domains.Agent.Bots.flow import FlowBot
        bot_class = FlowBot
    elif config.architecture_type == "multimodal":
        from app.Domains.Agent.Bots.multimodal import MultimodalBot
        bot_class = MultimodalBot
    else:
        from app.Domains.Agent.Bots.simple import SimpleBot
        bot_class = SimpleBot

    async def main():
        room_url = args.room_url
        token = args.token
        
        # Auto-create room if not provided
        if not room_url or not token:
            print("🔄 No room URL/token provided, creating Daily room...")
            room_url, token = await create_daily_room()
        
        system_messages = None
        
        # Determine base system prompt
        base_system_prompt = None
        if args.system_prompt:
             base_system_prompt = args.system_prompt
        elif loaded_assistant and loaded_assistant.agent.system_prompt:
             base_system_prompt = loaded_assistant.agent.system_prompt
        
        # Apply variables if provided
        final_system_prompt = base_system_prompt
        if args.prompt_variables and base_system_prompt:
            try:
                variables = json.loads(args.prompt_variables)
                final_system_prompt = base_system_prompt.format(**variables)
                print(f"🎨 Applied prompt variables: {variables}")
            except Exception as e:
                print(f"⚠️ Failed to apply prompt variables: {e}")

        if final_system_prompt:
            system_messages = [{"role": "system", "content": final_system_prompt}]

        webhook_config = None
        if loaded_assistant and loaded_assistant.webhooks:
            webhook_config = loaded_assistant.webhooks

        await run_bot(bot_class, config, room_url=room_url, token=token, system_messages=system_messages, webhook_config=webhook_config)
    
    asyncio.run(main())

if __name__ == "__main__":
    cli()
