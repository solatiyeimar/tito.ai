#!/usr/bin/env python3
"""Asterisk Runner - Start voice AI bot for telephony.

This runner creates a WebSocket server that accepts connections from
Asterisk's chan_websocket module and handles voice AI conversations.

Usage:
    # Start with default settings
    uv run python asterisk_runner.py
    
    # Specify host and port
    uv run python asterisk_runner.py --host 0.0.0.0 --port 8765
    
    # With specific providers
    uv run python asterisk_runner.py -l google -s deepgram -p cartesia

Asterisk Configuration:
    ;;; /etc/asterisk/extensions.conf
    [ai-agents]
    exten => 100,1,Answer()
    same => n,Set(JITTERBUFFER(adaptive)=60,300,40)
    same => n,Dial(WebSocket/pipecat/c(slin16)f(json))
    same => n,Hangup()
    
    ;;; /etc/asterisk/websocket_client.conf
    [pipecat]
    type = websocket_client
    uri = ws://YOUR_SERVER_IP:8765
    protocols = media
    connection_type = per_call_config
    connection_timeout = 1000
    reconnect_interval = 1000
    reconnect_attempts = 5
    tls_enabled = no
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


def setup_logging():
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="DEBUG" if os.getenv("DEBUG") else "INFO",
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Asterisk Voice AI Bot Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("ASTERISK_HOST", "0.0.0.0"),
        help="Host to bind the WebSocket server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ASTERISK_PORT", "8765")),
        help="Port for WebSocket connections (default: 8765)"
    )
    
    # LLM configuration
    parser.add_argument(
        "-l", "--llm-provider",
        type=str.lower,
        choices=["google", "openai", "anthropic", "groq"],
        help="LLM provider"
    )
    parser.add_argument(
        "-m", "--llm-model",
        type=str,
        help="LLM model name"
    )
    
    # STT configuration
    parser.add_argument(
        "-s", "--stt-provider",
        type=str.lower,
        choices=["deepgram", "groq"],
        help="STT provider"
    )
    
    # TTS configuration
    parser.add_argument(
        "-p", "--tts-provider",
        type=str.lower,
        choices=["deepgram", "cartesia", "elevenlabs"],
        help="TTS provider"
    )
    parser.add_argument(
        "-v", "--tts-voice",
        type=str,
        help="TTS voice ID"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    setup_logging()
    args = parse_args()
    
    # Set environment variables from CLI args
    if args.llm_provider:
        os.environ["LLM_PROVIDER"] = args.llm_provider
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model
    if args.stt_provider:
        os.environ["STT_PROVIDER"] = args.stt_provider
    if args.tts_provider:
        os.environ["TTS_PROVIDER"] = args.tts_provider
    if args.tts_voice:
        os.environ["TTS_VOICE"] = args.tts_voice
    
    # Import after setting env vars
    from app.Core.Config.bot import BotConfig
    from app.Domains.Agent.Bots.simple import SimpleBot
    from app.Domains.Agent.Bots.flow import FlowBot
    from app.Domains.Agent.Bots.multimodal import MultimodalBot
    
    # Create config
    config = BotConfig()
    
    # Select bot implementation based on configuration
    logger.info(f"Initializing bot with architecture: {config.architecture_type}")
    
    if config.architecture_type == "multimodal":
        bot = MultimodalBot(config)
    elif config.architecture_type == "flow":
        bot = FlowBot(config)
    else:
        bot = SimpleBot(config)
    
    # Setup transport
    # Note: unified setup_asterisk_transport method added to BaseBot and MultimodalBot
    if hasattr(bot, "setup_asterisk_transport"):
        bot.setup_asterisk_transport(host=args.host, port=args.port)
    else:
        raise RuntimeError(f"Bot type {type(bot).__name__} does not support Asterisk transport")

    bot.create_pipeline()
    
    # Print startup info
    logger.info("=" * 60)
    logger.info("🎙️  Asterisk Voice AI Bot")
    logger.info("=" * 60)
    logger.info(f"WebSocket Server: ws://{args.host}:{args.port}")
    logger.info(f"LLM Provider: {config.llm_provider} ({config.llm_model})")
    logger.info(f"STT Provider: {config.stt_provider}")
    logger.info(f"TTS Provider: {config.tts_provider}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Asterisk Dialplan Example:")
    logger.info("  exten => 100,1,Answer()")
    logger.info("  same => n,Set(JITTERBUFFER(adaptive)=60,300,40)")
    logger.info(f"  same => n,Dial(WebSocket/pipecat/c(slin16)f(json))")
    logger.info("  same => n,Hangup()")
    logger.info("")
    logger.info("Waiting for calls...")
    logger.info("=" * 60)
    
    # Start the bot
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
