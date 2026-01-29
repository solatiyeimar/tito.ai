from typing import List, Dict, Optional
import os
import datetime
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.daily.transport import DailyTransport, DailyParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.frames.frames import TranscriptionFrame
from app.Domains.Agent.Transports.asterisk.transport import AsteriskWSServerTransport, AsteriskWSServerParams
from app.Domains.Agent.Transports.asterisk.serializer import AsteriskWsFrameSerializer

from app.Core.Config.bot import BotConfig
from app.Services.webhook_sender import WebhookSender
from app.Http.DTOs.schemas import WebhookConfig
from app.Utils.analysis import analyze_conversation_with_gemini
from app.Domains.Agent.Tools.telephony import TRANSFER_CALL_TOOL
from app.Domains.Agent.Tools.context import GET_SECURE_DATA_TOOL
import time
import os

class MultimodalBot:
    """Bot for handling native speech-to-speech (multimodal) models."""

    def __init__(self, config: BotConfig, system_messages: Optional[List[Dict[str, str]]] = None, webhook_config: Optional[WebhookConfig] = None):
        self.config = config
        self.webhook_sender = WebhookSender(webhook_config)
        self.system_messages = system_messages or [{"role": "system", "content": "You are a helpful voice assistant."}]
        
        self.context = LLMContext(self.system_messages)
        self.user_aggregator, self.assistant_aggregator = LLMContextAggregatorPair(self.context)
        
        # Initialize the multimodal service based on the provider
        self.service = self._init_multimodal_service(config)
        
        self.transport_params = DailyParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            audio_in_sample_rate=16000,  # Ultravox recommended sample rate
            audio_out_sample_rate=16000,  # Match input sample rate for consistency
            vad_analyzer=SileroVADAnalyzer(),
            vad_enabled=True,
        )
        
        self.transport: Optional[DailyTransport] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None

    def _init_multimodal_service(self, config: BotConfig):
        system_instruction = self.system_messages[0]["content"]
        
        # Basic list of tools (can be expanded)
        tools = [TRANSFER_CALL_TOOL, GET_SECURE_DATA_TOOL] 
        
        match config.llm_provider:
            case "google": # Gemini Multimodal Live
                from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
                return GeminiLiveLLMService(
                    api_key=config.google_api_key,
                    voice_id="Puck", 
                    system_instruction=system_instruction,
                    tools=[self.transfer_call, self.get_secure_data_simple] 
                )
            case "openai": # OpenAI Realtime
                from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
                return OpenAIRealtimeLLMService(
                    api_key=config.openai_api_key,
                    voice="alloy",
                    instructions=system_instruction,
                    tools=tools # Pass schema
                )
            case "aws": # AWS Nova Sonic
                from pipecat.services.aws import AWSNovaSonicAdapter
                return AWSNovaSonicAdapter(
                    access_key=config.aws_access_key_id,
                    secret_key=config.aws_secret_access_key,
                    region=config.aws_region,
                    system_instruction=system_instruction
                )
            case "ultravox":
                from pipecat.services.ultravox.llm import OneShotInputParams, UltravoxRealtimeLLMService
                from pipecat.adapters.schemas.tools_schema import ToolsSchema
                
                return UltravoxRealtimeLLMService(
                    params=OneShotInputParams(
                        api_key=config.ultravox_api_key,
                        model=config.llm_model,
                        voice=config.tts_voice,
                        system_prompt=system_instruction,
                        temperature=config.llm_temperature or 0.3, # Use config temp or default
                    ),
                    one_shot_selected_tools=ToolsSchema(standard_tools=[]) # Fixme: Add tools
                )
            case _:
                raise ValueError(f"Provider {config.llm_provider} does not support native multimodal mode in this demo.")

    async def transfer_call(self, destination: str, reason: str = "user request"):
        """Sip transfer simulation."""
        logger.info(f"Using tool: transfer_call -> {destination} ({reason})")
        await self.webhook_sender.send("call_transfer", {
             "destination": destination,
             "reason": reason,
             "timestamp": time.time()
        })
        # Simulate delay then hangup
        await asyncio.sleep(2)
        if self.task:
             await self.task.cancel()
        return "Transfer initiated."

    async def get_secure_data(self, function_name, tool_call_id, args, llm, context, result_callback):
        """Tool handler for secure data retrieval (Generic)."""
        msg = await self.get_secure_data_simple()
        
        # For OpenAI Realtime (which calls result_callback)
        if result_callback:
             await result_callback({"data": msg})
        
        return msg

    async def get_secure_data_simple(self):
        """Retrieves secure data using the injected secret token."""
        token = os.environ.get("CRM_SECRET_TOKEN")
        logger.info(f"Checking secret token: {token}")
        if token == "super-secret-123":
             return "Access Granted: VIP Customer List [Alice, Bob]"
        return f"Access Denied. Token was: {token}"


    async def setup_transport(self, url: str, token: str):
        self.transport = DailyTransport(url, token, self.config.bot_name, self.transport_params)

        @self.transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await self.webhook_sender.send("participant_left", {"participant": participant, "reason": reason})
            if self.task:
                await self.task.cancel()

        @self.transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await self.webhook_sender.send("participant_joined", {"participant": participant})

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
        
        # Note: We override self.transport (typed as DailyTransport optional) 
        # Ideally we should update type hint to Union[DailyTransport, AsteriskWSServerTransport]
        self.transport = AsteriskWSServerTransport(params)
        
        @self.transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"📞 Client connected: {client.remote_address}")
            await self.webhook_sender.send("participant_joined", {"participant": {"id": "asterisk_user"}})
        
        @self.transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"📴 Client disconnected: {client.remote_address}")

    def create_pipeline(self):
        if not self.transport:
            raise RuntimeError("Transport must be set up before creating pipeline")
            
        from pipecat.processors.filters.function_filter import FunctionFilter

        # Build pipeline based on provider
        processors = [self.transport.input()]
        
        if self.config.llm_provider == "ultravox":
            # Ultravox needs aggregators
            processors.extend([
                self.user_aggregator,
                self.service,
                self.assistant_aggregator
            ])
        else:
            # Others like Gemini/OpenAI mostly handle it internal/direct
            processors.append(self.service)
            
        async def transcription_webhook(frame):
            if isinstance(frame, TranscriptionFrame):
                await self.webhook_sender.send("transcription", {
                    "user_id": frame.user_id,
                    "text": frame.text,
                    "timestamp": frame.timestamp,
                    "is_final": True 
                })
            return True
            
        processors.append(self.transport.output())
        
        # Add webhook filter before output to caption frames (if any flow through)
        # Note: Implementation details of multimodal services vary on where they emit frames.
        # Ideally we place it after service.
        processors.insert(-1, FunctionFilter(filter=transcription_webhook))

        pipeline = Pipeline(processors)

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
        if not self.runner or not self.task:
            raise RuntimeError("Bot not properly initialized.")
            
        room_url = getattr(self.transport, "room_url", None)
        await self.webhook_sender.send("call_started", {"room_url": room_url})
        try:
            await self.runner.run(self.task)
        finally:
            # Post-call analysis
            analysis_result = {}
            try:
                messages = self.context.messages
                # Use gathered messages if available, otherwise check if we can reconstruct from somewhere
                # For now rely on context
                api_key = self.config.google_api_key
                if api_key and messages:
                     analysis_result = await analyze_conversation_with_gemini(api_key, messages)
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                
            await self.webhook_sender.send("call_ended", {
                "timestamp": datetime.datetime.now().isoformat(),
                "analysis": analysis_result
            })
