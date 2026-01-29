from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, UUID4, validator
import uuid
from datetime import datetime

# --- Knowledge Base Schema ---

class KnowledgeBaseConfig(BaseModel):
    enabled: bool = False
    provider: Literal["text_file", "pinecone", "pdf_url"] = "text_file"
    source_uri: Optional[str] = None
    index_name: Optional[str] = None
    namespace: Optional[str] = None
    description: Optional[str] = None
    retrieval_settings: Dict[str, Any] = Field(default_factory=dict)

# --- Agent Schema ---

class AgentConfig(BaseModel):
    provider: Literal["google", "openai", "anthropic", "groq", "together", "mistral", "aws", "ultravox"] = "google"
    model: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    system_prompt: str = "You are a helpful assistant."
    initial_messages: List[Dict[str, str]] = Field(default_factory=list)
    knowledge_base: Optional[KnowledgeBaseConfig] = None
    tools: List[Dict[str, Any]] = Field(default_factory=list)

# --- Pipeline Settings Schema ---

class VADParams(BaseModel):
    confidence: float = 0.5
    start_secs: float = 0.2
    stop_secs: float = 0.8
    min_silence_duration_ms: Optional[int] = None
    min_volume: float = 0.6

class VADConfig(BaseModel):
    provider: Literal["silero", "webbtc"] = "silero"
    params: VADParams = Field(default_factory=VADParams)

class PipelineSettings(BaseModel):
    vad: VADConfig = Field(default_factory=VADConfig)
    interruptibility: bool = True
    speak_first: bool = True

# --- IO Layer Schema ---

class IOProviderParams(BaseModel):
    # Generic params dict to capture provider-specific settings like 'encoding', 'sample_rate'
    pass

class TransportConfig(BaseModel):
    provider: Literal["daily", "twilio-websocket", "websocket"] = "daily"
    params: Dict[str, Any] = Field(default_factory=dict)

class STTConfig(BaseModel):
    provider: Literal["deepgram", "gladia", "assemblyai", "groq", "ultravox"] = "deepgram"
    model: Optional[str] = None
    language: str = "en"
    params: Dict[str, Any] = Field(default_factory=dict)
    # Helper to keep previous specialized fields accessible if needed, or map them to params
    enable_mute_filter: bool = False 

class TTSConfig(BaseModel):
    provider: Literal["deepgram", "cartesia", "elevenlabs", "rime", "playht", "openai", "azure", "ultravox"] = "cartesia"
    voice_id: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    # Mapping common high-level fields
    speed: float = 1.0
    language: str = "en"

class SipConfig(BaseModel):
    # Standard SIP/Telephony handling options
    amd_enabled: bool = False
    amd_action_on_machine: Literal["hangup", "leave_message", "continue"] = "hangup"
    default_transfer_number: Optional[str] = None
    # Authentication & Trunking (Ultravox/Daily SIP often handled via room tokens, but we allow config here)
    auth_token: Optional[str] = None
    sip_uri: Optional[str] = None
    caller_id: Optional[str] = None
    # Asterisk Integration: Custom Headers for context propagation (e.g. X-Asterisk-CallID)
    sip_headers: Dict[str, str] = Field(default_factory=dict)

class IOLayerConfig(BaseModel):
    # type: The transport channel (webrtc = browser/app, sip = telephony/twilio)
    type: Literal["webrtc", "sip"] = "webrtc"
    transport: TransportConfig = Field(default_factory=TransportConfig)
    stt: Optional[STTConfig] = None
    tts: Optional[TTSConfig] = None
    sip: SipConfig = Field(default_factory=SipConfig)

# --- Webhook Configuration ---

class WebhookConfig(BaseModel):
    url: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    events: List[Literal["call_started", "call_ended", "participant_joined", "participant_left", "transcription", "call_transfer", "function_called", "node_transition"]] = Field(
        default_factory=lambda: ["call_started", "call_ended"]
    )

# --- Main Assistant Configuration ---

class AssistantConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Assistant"
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    
    # architecture_type: The internal architecture (simple, flow, multimodal)
    architecture_type: Literal["simple", "flow", "multimodal"] = "flow"
    
    pipeline_settings: PipelineSettings = Field(default_factory=PipelineSettings)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    io_layer: IOLayerConfig = Field(default_factory=IOLayerConfig)
    webhooks: Optional[WebhookConfig] = None
    flow: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

# --- API Resources (HATEOAS) ---

class Link(BaseModel):
    href: str
    method: str
    rel: str

class AssistantResponse(AssistantConfig):
    links: List[Link] = Field(default_factory=list, alias="_links")

class CallRequest(BaseModel):
    assistant_id: str
    phone_number: Optional[str] = None # For SIP/Tel
    variables: Optional[Dict[str, Any]] = None
    secrets: Optional[Dict[str, str]] = None
    dynamic_vocabulary: Optional[List[str]] = None

class CallResponse(BaseModel):
    id: str
    status: str
    room_url: Optional[str] = None
    token: Optional[str] = None
    links: List[Link] = Field(default_factory=list, alias="_links")


    # --- Backward Compatibility Helpers ---
    # These helpers ensure that existing code like parsers/bot_config_parser.py 
    # (which might expect flattened properties) continues to function or is easeir to update.
    
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
    def tools(self) -> List[Dict[str, Any]]:
        return self.agent.tools

    @property
    def tts_provider(self) -> Optional[str]:
        return self.io_layer.tts.provider if self.io_layer.tts else None
        
    @property
    def tts_voice(self) -> Optional[str]:
        return self.io_layer.tts.voice_id if self.io_layer.tts else None
        
    @property
    def stt_provider(self) -> Optional[str]:
        return self.io_layer.stt.provider if self.io_layer.stt else None

    @property
    def enable_stt_mute_filter(self) -> bool:
        return self.io_layer.stt.enable_mute_filter

