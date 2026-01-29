import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# --- Sub-Models (Value Objects) ---


class KnowledgeBaseConfig(BaseModel):
    enabled: bool = False
    provider: Literal["text_file", "pinecone", "pdf_url"] = "text_file"
    source_uri: Optional[str] = None
    index_name: Optional[str] = None
    namespace: Optional[str] = None
    description: Optional[str] = None
    retrieval_settings: Dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    provider: Literal[
        "google", "openai", "anthropic", "groq", "together", "mistral", "aws", "ultravox"
    ] = "google"
    model: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    system_prompt: str = "You are a helpful assistant."
    initial_messages: List[Dict[str, str]] = Field(default_factory=list)
    knowledge_base: Optional[KnowledgeBaseConfig] = None
    tools: List[Dict[str, Any]] = Field(default_factory=list)


class VADParams(BaseModel):
    confidence: float = 0.5
    start_secs: float = 0.2
    stop_secs: float = 0.8
    min_silence_duration_ms: Optional[int] = None
    min_volume: float = 0.6


class VADConfig(BaseModel):
    provider: Literal["silero", "webbtc"] = "silero"
    params: VADParams = Field(default_factory=VADParams)


class InactivityMessage(BaseModel):
    message: str
    timeout: float = 10.0
    end_behavior: Literal["continue", "hangup"] = "continue"


class PipelineSettings(BaseModel):
    vad: VADConfig = Field(default_factory=VADConfig)
    interruptibility: bool = True
    speak_first: bool = True
    initial_message: Optional[str] = None
    initial_delay: float = 0.0
    initial_message_interruptible: bool = True
    inactivity_messages: List[InactivityMessage] = Field(default_factory=list)


class TransportConfig(BaseModel):
    provider: Literal["daily", "twilio-websocket", "websocket"] = "daily"
    params: Dict[str, Any] = Field(default_factory=dict)


class STTConfig(BaseModel):
    provider: Literal["deepgram", "gladia", "assemblyai", "groq", "ultravox"] = "deepgram"
    model: Optional[str] = None
    language: str = "en"
    params: Dict[str, Any] = Field(default_factory=dict)
    enable_mute_filter: bool = False


class TTSConfig(BaseModel):
    provider: Literal[
        "deepgram", "cartesia", "elevenlabs", "rime", "playht", "openai", "azure", "ultravox"
    ] = "cartesia"
    voice_id: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    speed: float = 1.0
    language: str = "en"


class SipConfig(BaseModel):
    amd_enabled: bool = False
    amd_action_on_machine: Literal["hangup", "leave_message", "continue"] = "hangup"
    default_transfer_number: Optional[str] = None
    auth_token: Optional[str] = None
    sip_uri: Optional[str] = None
    caller_id: Optional[str] = None
    sip_headers: Dict[str, str] = Field(default_factory=dict)


class IOLayerConfig(BaseModel):
    type: Literal["webrtc", "sip"] = "webrtc"
    transport: TransportConfig = Field(default_factory=TransportConfig)
    stt: Optional[STTConfig] = None
    tts: Optional[TTSConfig] = None
    sip: SipConfig = Field(default_factory=SipConfig)


class WebhookConfig(BaseModel):
    url: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    events: List[str] = Field(default_factory=lambda: ["call_started", "call_ended"])


# --- Aggregate Root ---


class Assistant(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Assistant"
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    architecture_type: Literal["simple", "flow", "multimodal"] = "flow"
    pipeline_settings: PipelineSettings = Field(default_factory=PipelineSettings)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    io_layer: IOLayerConfig = Field(default_factory=IOLayerConfig)
    webhooks: Optional[WebhookConfig] = None
    flow: Optional[Dict[str, Any]] = None

    # Backward compatibility properties for simple access
    @property
    def system_prompt(self) -> str:
        return self.agent.system_prompt

    @property
    def llm_provider(self) -> str:
        return self.agent.provider

    @property
    def llm_model(self) -> Optional[str]:
        return self.agent.model

    @property
    def llm_temperature(self) -> float:
        return self.agent.temperature

    @property
    def stt_provider(self) -> Optional[str]:
        return self.io_layer.stt.provider if self.io_layer.stt else None

    @property
    def tts_provider(self) -> Optional[str]:
        return self.io_layer.tts.provider if self.io_layer.tts else None

    @property
    def tts_voice(self) -> Optional[str]:
        return self.io_layer.tts.voice_id if self.io_layer.tts else None
