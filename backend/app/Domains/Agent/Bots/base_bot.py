"""Base bot framework for shared functionality."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.filters.stt_mute_filter import (
    STTMuteFilter,
    STTMuteConfig,
    STTMuteStrategy,
)
from pipecat.services.rime.tts import RimeHttpTTSService
from pipecat.transports.daily.transport import DailyTransport, DailyParams, VADParams
from app.Domains.Agent.Transports.asterisk.transport import AsteriskWSServerTransport, AsteriskWSServerParams
from app.Domains.Agent.Transports.asterisk.serializer import AsteriskWsFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.utils.sync.event_notifier import EventNotifier
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.frames.frames import (
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
    LLMContextFrame,
    LLMMessagesFrame,
    StartInterruptionFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
)
from deepgram import LiveOptions

import os
import asyncio
from loguru import logger
import time
from app.Services.webhook_sender import WebhookSender
from app.Http.DTOs.schemas import WebhookConfig
from app.Utils.analysis import analyze_conversation_with_gemini

# from .smart_endpointing import (
#     CLASSIFIER_SYSTEM_INSTRUCTION,
#     CompletenessCheck,
#     OutputGate,
#     StatementJudgeContextFilter,
# )


class BaseBot(ABC):
    """Abstract base class for bots, providing core Pipecat integration."""

    def __init__(self, config, system_messages: List[Dict[str, str]], webhook_config: Optional[WebhookConfig] = None):
        """Initialize the bot with services and common components."""
        self.config = config
        
        # Initialize context aggregator
        self.context = LLMContext(messages=system_messages)
        self.context_aggregator = LLMContextAggregatorPair(self.context)
        
        # Initialize services
        self.stt = self._init_stt(config)
        self.tts = self._init_tts(config)
        self.llm = self._init_llm(config, system_messages)
        
        self.stt_mute_filter = STTMuteFilter(
            config=STTMuteConfig(
                strategies={STTMuteStrategy.ALWAYS},
            )
        )
        
        # Initialize RTVI with default config
        self.rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        async def on_user_idle(frame):
            pass

        self.user_idle = UserIdleProcessor(callback=on_user_idle, timeout=5.0)
        
        # These will be set up when needed
        self.transport: Optional[DailyTransport] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        
        self.webhook_sender = WebhookSender(webhook_config)

        logger.debug(f"Initialised bot with config: {config}")

    def _init_stt(self, config):
        match config.stt_provider:
            case "deepgram":
                from pipecat.services.deepgram.stt import DeepgramSTTService
                from deepgram import LiveOptions
                keywords = []
                stt_keywords_env = os.getenv("STT_KEYWORDS")
                if stt_keywords_env:
                    keywords = stt_keywords_env.split(",")
                    
                # Define a custom class to support detect_language and safe finalize
                class CustomLiveOptions:
                    def __init__(self, **kwargs):
                        self._kwargs = kwargs
                    def to_dict(self):
                        return self._kwargs
                    @property
                    def sample_rate(self):
                         return self._kwargs.get("sample_rate")
                    @property
                    def model(self):
                         return self._kwargs.get("model")
                    def __getattr__(self, name):
                         return self._kwargs.get(name)

                class CustomDeepgramSTTService(DeepgramSTTService):
                    async def process_frame(self, frame, direction):
                        await super(DeepgramSTTService, self).process_frame(frame, direction)

                        if isinstance(frame, UserStartedSpeakingFrame) and not self.vad_enabled:
                            await self.start_metrics()
                        elif isinstance(frame, UserStoppedSpeakingFrame):
                            # Safe finalize
                            try:
                                if self._connection and await self._connection.is_connected():
                                    await self._connection.finalize()
                                    logger.trace(f"Triggered finalize event on: {frame.name=}, {direction=}")
                            except Exception:
                                pass

                # Configure language detection if 'multi' or 'auto'
                detect_language = config.stt_language in ("multi", "auto")
                language = None if detect_language else config.stt_language

                options = {
                    "model": config.stt_model,
                    "smart_format": True,
                    "interim_results": True,
                    "endpointing": 300,
                }
                
                if keywords:
                    options["keywords"] = keywords

                if language:
                    options["language"] = language
                
                if detect_language:
                    options["detect_language"] = True
                
                logger.info(f"Initialized Deepgram with options: {options}")

                return CustomDeepgramSTTService(
                    api_key=config.deepgram_api_key,
                    live_options=CustomLiveOptions(**options)
                )
            case _:
                raise ValueError(f"Invalid STT provider: {config.stt_provider}")

    def _init_tts(self, config):
        match config.tts_provider:
            case "cartesia":
                from pipecat.services.cartesia.tts import CartesiaTTSService
                return CartesiaTTSService(
                    api_key=config.cartesia_api_key,
                    voice_id=config.tts_voice,
                )
            case "elevenlabs":
                from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
                return ElevenLabsTTSService(
                    api_key=config.elevenlabs_api_key,
                    voice_id=config.tts_voice,
                )
            case "deepgram":
                from pipecat.services.deepgram.tts import DeepgramTTSService
                return DeepgramTTSService(
                    api_key=config.deepgram_api_key,
                    voice=config.tts_voice,
                )
            case "rime":
                from pipecat.services.rime.tts import RimeHttpTTSService
                return RimeHttpTTSService(
                    api_key=config.rime_api_key,
                    voice_id=config.tts_voice,
                )
            case "playht":
                from pipecat.services.playht.tts import PlayHTTTSService
                return PlayHTTTSService(
                    api_key=config.playht_api_key,
                    user_id=config.playht_user_id,
                    voice_url=config.tts_voice,
                )
            case "openai":
                from pipecat.services.openai.tts import OpenAITTSService
                return OpenAITTSService(
                    api_key=config.openai_api_key,
                    voice=config.tts_voice,
                )
            case "azure":
                from pipecat.services.azure.tts import AzureTTSService
                return AzureTTSService(
                    api_key=config.azure_api_key,
                    region=config.azure_region,
                    voice=config.tts_voice,
                )
            case _:
                raise ValueError(f"Invalid TTS provider: {config.tts_provider}")

    def _init_llm(self, config, system_messages):
        system_instruction = system_messages[0]["content"] if system_messages else "You are a voice assistant"
        
        match config.llm_provider:
            case "google":
                from pipecat.services.google.llm import GoogleLLMService
                return GoogleLLMService(
                    api_key=config.google_api_key,
                    model=config.llm_model,
                    params=config.google_params,
                    system_instruction=system_instruction,
                    tools=config.tools
                )
            case "openai":
                from pipecat.services.openai.llm import OpenAILLMService
                service = OpenAILLMService(
                    api_key=config.openai_api_key,
                    model=config.llm_model,
                    params=config.openai_params
                )
                return service
            case "anthropic":
                from pipecat.services.anthropic.llm import AnthropicLLMService
                return AnthropicLLMService(
                    api_key=config.anthropic_api_key,
                    model=config.llm_model
                )
            case "groq":
                from pipecat.services.groq.llm import GroqLLMService
                return GroqLLMService(
                    api_key=config.groq_api_key,
                    model=config.llm_model
                )
            case "together":
                from pipecat.services.together import TogetherLLMService
                return TogetherLLMService(
                    api_key=config.together_api_key,
                    model=config.llm_model
                )
            case "mistral":
                from pipecat.services.mistral import MistralLLMService
                return MistralLLMService(
                    api_key=config.mistral_api_key,
                    model=config.llm_model
                )
            case "ultravox":
                from pipecat.services.ultravox.llm import UltravoxRealtimeLLMService, OneShotInputParams
                return UltravoxRealtimeLLMService(
                    params=OneShotInputParams(
                        api_key=config.ultravox_api_key,
                        model=config.llm_model,
                        voice=config.tts_voice,
                        system_prompt=system_instruction,
                        temperature=config.llm_temperature or 0.3
                    )
                )
            case _:
                raise ValueError(f"Invalid LLM provider: {config.llm_provider}")

    async def setup_transport(self, url: str, token: str):
        """Set up the transport and its internal event handlers."""
        # Standard configuration for Daily transport
        transport_params = DailyParams(
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_params=VADParams(
                confidence=0.7,
                start_secs=0.2,
                stop_secs=0.8,
                min_volume=0.6,
            ),
        )

        self.transport = DailyTransport(
            room_url=url,
            token=token,
            bot_name=self.config.bot_name,
            params=transport_params,
        )

        @self.transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            logger.info(f"Participant joined: {participant['id']}")
            await self.webhook_sender.send("participant_joined", {"participant": participant})
            if hasattr(transport, "capture_participant_transcription"):
                await transport.capture_participant_transcription(participant["id"])
            # The first participant (user) triggers the bot's greeting/start
            await self._handle_first_participant()

        @self.transport.event_handler("on_app_message")
        async def on_app_message(transport, message, sender):
            if "message" not in message:
                return
            
            # TODO: Handle app messages if needed via webhooks?

            await self.task.queue_frames(
                [
                    UserStartedSpeakingFrame(),
                    TranscriptionFrame(
                        user_id=sender, timestamp=time.time(), text=message["message"]
                    ),
                    UserStoppedSpeakingFrame(),
                ]
            )

    def setup_asterisk_transport(self, host: str, port: int):
        """Set up the Asterisk WebSocket transport."""
        params = AsteriskWSServerParams(
            host=host,
            port=port,
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=AsteriskWsFrameSerializer(),
        )
        
        self.transport = AsteriskWSServerTransport(params)
        
        # Register standard event handlers mapping to bot logic
        @self.transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"📞 Client connected: {client.remote_address}")
            # Map Asterisk connection to first participant logic (starts conversation)
            await self._handle_first_participant()
        
        @self.transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"📴 Client disconnected: {client.remote_address}")

    async def handle_dtmf(self, digit: str, call_id: str):
        """Handle DTMF digit received during call (optional override)."""
        logger.info(f"DTMF received: {digit} (call: {call_id})")

    def create_pipeline(self):
        """Create the processing pipeline."""
        if not self.transport:
            raise RuntimeError("Transport must be set up before creating pipeline")

        # Register tools if available in config
        if self.config.tools:
            self._register_tools(self.config.tools)

        async def transcription_webhook(frame):
            if isinstance(frame, TranscriptionFrame):
                await self.webhook_sender.send("transcription", {
                    "user_id": frame.user_id,
                    "text": frame.text,
                    "timestamp": frame.timestamp,
                    "is_final": True # Pipecat transcription frames are usually final segments
                })
            return True

        # Build pipeline with Deepgram STT at the beginning
        pipeline = Pipeline(
            [
                processor
                for processor in [
                    self.rtvi,
                    self.transport.input(),
                    self.stt_mute_filter,
                    self.stt,  # Deepgram transcribes incoming audio
                    FunctionFilter(filter=transcription_webhook), # Hook for transcription webhooks
                    self.context_aggregator.user(),
                    self.llm,
                    self.tts,
                    self.user_idle,
                    self.transport.output(),
                    self.context_aggregator.assistant(),
                ]
                if processor is not None
            ]
        )

        self.task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )
        self.runner = PipelineRunner()

    async def start(self):
        """Start the bot's main task."""
        if not self.runner or not self.task:
            raise RuntimeError("Bot not properly initialized. Call create_pipeline first.")
        
        room_url = getattr(self.transport, "room_url", None)
        await self.webhook_sender.send("call_started", {"room_url": room_url})
        await self.runner.run(self.task)

    async def cleanup(self):
        """Clean up resources and analyze call."""
        if self.runner:
            await self.runner.stop_when_done()
        if self.transport:
            await self.transport.close()
        
        # Perform Post-Call Analysis
        analysis_result = {}
        try:
            # Gather messages from context
            messages = self.context.messages
            # Prefer Google key, fallback to others if needed, using Gemini for cost/speed
            api_key = self.config.google_api_key
            if api_key and messages:
                logger.info("Running post-call analysis...")
                analysis_result = await analyze_conversation_with_gemini(api_key, messages)
        except Exception as e:
            logger.error(f"Analysis error: {e}")

        await self.webhook_sender.send("call_ended", {
            "timestamp": time.time(),
            "analysis": analysis_result
        })

    # --- Tool Call Handlers ---

    def _register_tools(self, tool_schemas: List[Dict]):
        """Register tools with the LLM service."""
        for tool in tool_schemas:
            # Extract function name from different schema formats (OpenAI vs Google)
            if "function" in tool:
                func_name = tool["function"]["name"]
            elif "name" in tool:
                 func_name = tool["name"]
            else:
                 continue

            # Look for a specific handler method, fallback to generic
            handler_name = f"handle_{func_name}"
            handler = getattr(self, handler_name, self.generic_tool_handler)
            logger.debug(f"Registering tool handler: {func_name} -> {handler.__name__}")
            # Note: For OpenAI/Google standard services, we use register_function
            self.llm.register_function(func_name, handler)

    async def handle_transfer_call(self, function_name, tool_call_id, args, llm, context, result_callback):
        """Standard handler for call transfers."""
        destination = args.get("destination", "Operator")
        reason = args.get("reason", "Not specified")
        logger.info(f"☎\ufe0f Tool Call: Transferring to {destination}. Reason: {reason}")
        
        await self.webhook_sender.send("call_transfer", {
            "destination": destination,
            "reason": reason
        })
        
        # In a real telephony transport, we would call self.transport.transfer_call()
        # For now, we simulate the action
        if result_callback:
            await result_callback({"status": "transfer_initiated", "destination": destination})

    async def generic_tool_handler(self, function_name, tool_call_id, args, llm, context, result_callback):
        """Default handler for any tools without a dedicated method."""
        logger.info(f"🛠\ufe0f Tool Call: {function_name} with args {args}")
        if result_callback:
            await result_callback({"status": "success", "message": f"Processed {function_name}"})


    @abstractmethod
    async def _handle_first_participant(self):
        """Override in subclass to handle the first participant joining."""
        pass
