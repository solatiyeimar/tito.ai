"""Base bot framework for shared functionality."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.filters.stt_mute_filter import (
    STTMuteConfig,
    STTMuteFilter,
    STTMuteStrategy,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.transports.daily.transport import DailyParams, DailyTransport, VADParams

from app.Domains.Agent.Factory.service_factory import ServiceFactory
from app.Domains.Agent.Transports.asterisk.serializer import AsteriskWsFrameSerializer
from app.Domains.Agent.Transports.asterisk.transport import (
    AsteriskWSServerParams,
    AsteriskWSServerTransport,
)
from app.Http.DTOs.schemas import WebhookConfig
from app.Services.webhook_sender import WebhookSender
from app.Utils.analysis import analyze_conversation_with_gemini


class BaseBot(ABC):
    """Abstract base class for bots, providing core Pipecat integration."""

    def __init__(
        self,
        config,
        system_messages: List[Dict[str, str]],
        webhook_config: Optional[WebhookConfig] = None,
    ):
        """Initialize the bot with services and common components."""
        self.config = config

        # Initialize context aggregator
        self.context = LLMContext(messages=system_messages)
        self.context_aggregator = LLMContextAggregatorPair(self.context)

        # Initialize services using Factory
        self.stt = ServiceFactory.create_stt_service(config)
        self.tts = ServiceFactory.create_tts_service(config)
        self.llm = ServiceFactory.create_llm_service(config, system_messages)

        self.stt_mute_filter = STTMuteFilter(
            config=STTMuteConfig(
                strategies={STTMuteStrategy.ALWAYS},
            )
        )

        # Initialize RTVI with default config
        self.rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # Idle state
        self.idle_stage = 0

        async def on_user_idle(frame):
            if not self.config.inactivity_messages:
                return

            if self.idle_stage >= len(self.config.inactivity_messages):
                return

            stage_config = self.config.inactivity_messages[self.idle_stage]
            message = stage_config.get("message")
            behavior = stage_config.get("end_behavior", "continue")

            logger.info(
                f"User idle stage {self.idle_stage}. Action: {behavior}. Message: {message}"
            )

            if message:
                # Prompt LLM to speak the message
                messages = list(self.context.messages)
                messages.append(
                    {
                        "role": "system",
                        "content": f"The user is silent. Say exactly this to re-engage: '{message}'",
                    }
                )
                if self.task:
                    # Queue message update AND trigger LLM run
                    await self.task.queue_frames([LLMMessagesUpdateFrame(messages=messages), LLMRunFrame()])

            if behavior == "hangup":
                logger.info("Ending call due to inactivity.")
                if self.task:
                    from pipecat.frames.frames import EndFrame
                    await asyncio.sleep(1.0)
                    await self.task.queue_frames([EndFrame()])
                return

            # Advance stage
            self.idle_stage += 1
            if self.idle_stage < len(self.config.inactivity_messages):
                next_timeout = self.config.inactivity_messages[self.idle_stage].get("timeout", 10.0)
                if self.user_idle:
                    self.user_idle.timeout = next_timeout
            else:
                # No more stages
                if self.user_idle:
                    self.user_idle.timeout = 3600.0

        # Configure User Idle Processor
        if self.config.inactivity_messages:
            first_timeout = self.config.inactivity_messages[0].get("timeout", 10.0)
            self.user_idle = UserIdleProcessor(callback=on_user_idle, timeout=first_timeout)
        else:
            self.user_idle = None

        # These will be set up when needed
        self.transport: Optional[DailyTransport] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None

        self.webhook_sender = WebhookSender(webhook_config)

        self.appointments = []

        logger.debug(f"Initialised bot with config: {config}")

    async def setup_transport(self, url: str, token: str):
        """Set up the transport and its internal event handlers."""
        # Standard configuration for Daily transport
        transport_params = DailyParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7,
                    start_secs=0.2,
                    stop_secs=0.8,
                    min_volume=0.6,
                )
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

    async def _reset_idle_monitor_if_needed(self, frame):
        """Reset idle stage when user speaks."""
        if isinstance(frame, UserStartedSpeakingFrame):
            self.idle_stage = 0
            if self.user_idle and self.config.inactivity_messages:
                first_timeout = self.config.inactivity_messages[0].get("timeout", 10.0)
                self.user_idle.timeout = first_timeout
        return True

    def create_pipeline(self):
        """Create the processing pipeline."""
        if not self.transport:
            raise RuntimeError("Transport must be set up before creating pipeline")

        # Register tools if available in config
        if self.config.tools:
            self._register_tools(self.config.tools)

        async def transcription_webhook(frame):
            if isinstance(frame, TranscriptionFrame):
                await self.webhook_sender.send(
                    "transcription",
                    {
                        "user_id": frame.user_id,
                        "text": frame.text,
                        "timestamp": frame.timestamp,
                        "is_final": True,  # Pipecat transcription frames are usually final segments
                    },
                )
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
                    FunctionFilter(filter=transcription_webhook),  # Hook for transcription webhooks
                    FunctionFilter(filter=self._reset_idle_monitor_if_needed),
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
                allow_interruptions=self.config.interruptibility,
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
            # DailyTransport handles cleanup via the runner/client interaction
            pass

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

        await self.webhook_sender.send(
            "call_ended", {"timestamp": time.time(), "analysis": analysis_result}
        )

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

    async def handle_transfer_call(self, params: FunctionCallParams):
        """Standard handler for call transfers."""
        args = params.arguments
        destination = args.get("destination", "Operator")
        reason = args.get("reason", "Not specified")
        logger.info(f"☎\ufe0f Tool Call: Transferring to {destination}. Reason: {reason}")

        await self.webhook_sender.send(
            "call_transfer", {"destination": destination, "reason": reason}
        )

        # In a real telephony transport, we would call self.transport.transfer_call()
        # For now, we simulate the action
        if params.result_callback:
            await params.result_callback({"status": "transfer_initiated", "destination": destination})

    async def handle_create_crm_lead(self, params: FunctionCallParams):
        """Mock handler for creating a CRM lead."""
        logger.info(f"🛠️ Tool Call: create_crm_lead {params.arguments}")
        if params.result_callback:
            await params.result_callback({"status": "success", "message": "Lead created successfully in CRM."})

    async def handle_search_customer(self, params: FunctionCallParams):
        """Mock handler for searching a customer."""
        logger.info(f"🛠️ Tool Call: search_customer {params.arguments}")
        if params.result_callback:
            await params.result_callback({"status": "success", "message": "Customer found: Juan Perez (ID: 12345)."})

    async def handle_schedule_appointment(self, params: FunctionCallParams):
        """Handler for scheduling an appointment (Session Memory)."""
        logger.info(f"🛠️ Tool Call: schedule_appointment {params.arguments}")
        args = params.arguments
        
        # Store in session memory
        appointment = {
            "datetime": args.get("datetime"),
            "customer_id": args.get("customer_id"),
            "status": "confirmed"
        }
        self.appointments.append(appointment)
        
        if params.result_callback:
            await params.result_callback({"status": "success", "message": f"Appointment scheduled for {appointment['datetime']}."})

    async def handle_get_scheduled_appointments(self, params: FunctionCallParams):
        """Handler for retrieving scheduled appointments (Session Memory)."""
        logger.info(f"🛠️ Tool Call: get_scheduled_appointments")
        
        if not self.appointments:
            message = "No appointments scheduled in this session."
        else:
            appt_list = ", ".join([f"{a['datetime']} (Customer: {a['customer_id']})" for a in self.appointments])
            message = f"Appointments in this session: {appt_list}"

        if params.result_callback:
            await params.result_callback({"appointments": self.appointments, "message": message})

    async def generic_tool_handler(self, params: FunctionCallParams):
        """Default handler for any tools without a dedicated method."""
        logger.info(f"🛠\ufe0f Tool Call: {params.function_name} with args {params.arguments}")
        if params.result_callback:
            await params.result_callback({"status": "success", "message": f"Processed {params.function_name}"})

    @abstractmethod
    async def _handle_first_participant(self):
        """Override in subclass to handle the first participant joining."""
        pass
