"""Asterisk Bot - Voice AI bot for telephony via Asterisk.

This bot handles phone calls through Asterisk's chan_websocket module,
providing the same AI capabilities as the Daily-based bots.
"""

import os
from typing import Optional

from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.audio.vad.silero import SileroVADAnalyzer

from app.Core.Config.bot import BotConfig
from app.Domains.Agent.Transports.asterisk.transport import (
    AsteriskWSTransport,
    AsteriskWSParams,
    CallStartFrame,
    DTMFFrame,
    HangupFrame,
)


class AsteriskBot:
    """Voice AI bot for Asterisk telephony.
    
    This bot connects to Asterisk via WebSocket and provides:
    - Speech-to-Text (STT)
    - LLM-based conversation
    - Text-to-Speech (TTS)
    - DTMF handling for IVR functionality
    """
    
    def __init__(self, config: BotConfig):
        """Initialize the Asterisk bot.
        
        Args:
            config: Bot configuration instance.
        """
        self.config = config
        
        # System messages for LLM
        self.system_messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            }
        ]
        
        # Import and initialize services
        self._init_services(config)
        
        # Pipeline components
        self.transport: Optional[AsteriskWSTransport] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are a helpful voice assistant handling phone calls.
        
Important guidelines:
- Keep responses concise and clear for phone conversations
- Speak naturally as if on a phone call
- Ask clarifying questions when needed
- Be patient with audio quality issues
- Confirm important information back to the caller
"""
    
    def _init_services(self, config: BotConfig):
        """Initialize STT, LLM, and TTS services."""
        # Import STT
        match config.stt_provider:
            case "deepgram":
                from pipecat.services.deepgram.stt import DeepgramSTTService
                self.stt = DeepgramSTTService(
                    api_key=config.deepgram_api_key,
                    # Use 8kHz for telephony if possible
                )
            case "groq":
                from pipecat.services.groq.stt import GroqSTTService
                self.stt = GroqSTTService(api_key=config.groq_api_key)
            case _:
                from pipecat.services.deepgram.stt import DeepgramSTTService
                self.stt = DeepgramSTTService(api_key=config.deepgram_api_key)
        
        # Import LLM
        match config.llm_provider:
            case "google":
                from pipecat.services.google.llm import GoogleLLMService
                self.llm = GoogleLLMService(
                    api_key=config.google_api_key,
                    model=config.llm_model,
                )
            case "openai":
                from pipecat.services.openai.llm import OpenAILLMService
                self.llm = OpenAILLMService(
                    api_key=config.openai_api_key,
                    model=config.llm_model,
                )
            case "anthropic":
                from pipecat.services.anthropic.llm import AnthropicLLMService
                self.llm = AnthropicLLMService(
                    api_key=config.anthropic_api_key,
                    model=config.llm_model,
                )
            case "groq":
                from pipecat.services.groq.llm import GroqLLMService
                self.llm = GroqLLMService(
                    api_key=config.groq_api_key,
                    model=config.llm_model,
                )
            case "ultravox":
                from pipecat.services.ultravox.llm import UltravoxRealtimeLLMService, OneShotInputParams
                self.llm = UltravoxRealtimeLLMService(
                    params=OneShotInputParams(
                        api_key=config.ultravox_api_key,
                        model=config.llm_model,
                        system_prompt=self._get_system_prompt(),
                        temperature=config.llm_temperature
                    )
                )
            case _:
                from pipecat.services.google.llm import GoogleLLMService
                self.llm = GoogleLLMService(
                    api_key=config.google_api_key,
                    model=config.llm_model,
                )
        
        # Import TTS
        match config.tts_provider:
            case "deepgram":
                from pipecat.services.deepgram.tts import DeepgramTTSService
                self.tts = DeepgramTTSService(
                    api_key=config.deepgram_api_key,
                    voice=config.tts_voice,
                )
            case "cartesia":
                from pipecat.services.cartesia.tts import CartesiaTTSService
                self.tts = CartesiaTTSService(
                    api_key=config.cartesia_api_key,
                    voice_id=config.tts_voice,
                )
            case "elevenlabs":
                from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
                self.tts = ElevenLabsTTSService(
                    api_key=config.elevenlabs_api_key,
                    voice_id=config.tts_voice,
                )
            case _:
                from pipecat.services.deepgram.tts import DeepgramTTSService
                self.tts = DeepgramTTSService(
                    api_key=config.deepgram_api_key,
                    voice=config.tts_voice,
                )
        
        # Initialize LLM context
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
        )
        self.context = LLMContext(messages=self.system_messages)
        self.context_aggregator = LLMContextAggregatorPair(self.context)
    
    def setup_transport(self, host: str = "0.0.0.0", port: int = 8765):
        """Set up the Asterisk WebSocket transport.
        
        Args:
            host: Host to bind the server.
            port: Port for WebSocket connections.
        """
        params = AsteriskWSParams(
            host=host,
            port=port,
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
        
        self.transport = AsteriskWSTransport(params)
        
        # Register event handlers
        @self.transport.event_handler("on_call_start")
        async def on_call_start(transport, call_id):
            logger.info(f"📞 Call started: {call_id}")
        
        @self.transport.event_handler("on_call_end")
        async def on_call_end(transport, call_id):
            logger.info(f"📴 Call ended: {call_id}")
    
    def create_pipeline(self):
        """Create the processing pipeline."""
        if not self.transport:
            raise RuntimeError("Transport not set up. Call setup_transport() first.")
        
        pipeline = Pipeline([
            self.transport.input(),
            self.stt,
            self.context_aggregator.user(),
            self.llm,
            self.tts,
            self.transport.output(),
        ])
        
        self.task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            )
        )
        
        self.runner = PipelineRunner()
    
    async def start(self):
        """Start the bot server."""
        if not self.transport:
            raise RuntimeError("Transport not set up. Call setup_transport() first.")
        
        logger.info("🚀 Starting Asterisk bot server...")
        await self.transport.start_server()
    
    async def handle_dtmf(self, digit: str, call_id: str):
        """Handle DTMF digit received during call.
        
        Override this method to implement IVR functionality.
        
        Args:
            digit: The DTMF digit (0-9, *, #, A-D)
            call_id: The call session ID
        """
        logger.info(f"DTMF received: {digit} (call: {call_id})")
        # Default: no special handling
        # Override in subclass for IVR functionality
