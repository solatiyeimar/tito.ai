"""Bot configuration management module."""

import os
from typing import TypedDict, Literal, NotRequired, List, Dict, Any, Optional
from dotenv import load_dotenv
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import BaseOpenAILLMService


class DailyConfig(TypedDict):
    api_key: str
    api_url: str
    room_url: NotRequired[str]


BotType = Literal["simple", "flow", "multimodal"]


class BotConfig:
    def __init__(self):
        load_dotenv()
        
        self.tools: List[Dict[str, Any]] = []
        self.flow_config: Optional[Any] = None

        # Validate core required vars
        required = {
            "DAILY_API_KEY": os.getenv("DAILY_API_KEY"),
        }

        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing basic required env vars: {', '.join(missing)}")

        self.daily: DailyConfig = {
            "api_key": required["DAILY_API_KEY"],
            "api_url": os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        }

        # Bot configuration
        self._bot_type: BotType = os.getenv("ARCHITECTURE_TYPE", "flow")
        if self._bot_type not in ("simple", "flow", "multimodal"):
            self._bot_type = "flow"

    def __repr__(self) -> str:
        return f"BotConfig(architecture_type={self.architecture_type}, bot_name={self.bot_name}, llm_provider={self.llm_provider}, tts_provider={self.tts_provider})"

    def _is_truthy(self, value: str) -> bool:
        if not value:
            return False
        return value.lower() in (
            "true",
            "1",
            "t",
            "yes",
            "y",
            "on",
            "enable",
            "enabled",
            "si",
            "ok",
            "okay",
        )

    ###########################################################################
    # API keys
    ###########################################################################

    @property
    def google_api_key(self) -> str:
        return os.getenv("GOOGLE_API_KEY")

    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY")

    @property
    def deepgram_api_key(self) -> str:
        return os.getenv("DEEPGRAM_API_KEY")

    @property
    def cartesia_api_key(self) -> str:
        return os.getenv("CARTESIA_API_KEY")

    @property
    def elevenlabs_api_key(self) -> str:
        return os.getenv("ELEVENLABS_API_KEY")

    @property
    def anthropic_api_key(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def groq_api_key(self) -> str:
        return os.getenv("GROQ_API_KEY")

    @property
    def together_api_key(self) -> str:
        return os.getenv("TOGETHER_API_KEY")

    @property
    def mistral_api_key(self) -> str:
        return os.getenv("MISTRAL_API_KEY")

    @property
    def playht_api_key(self) -> str:
        return os.getenv("PLAYHT_API_KEY")

    @property
    def playht_user_id(self) -> str:
        return os.getenv("PLAYHT_USER_ID")

    @property
    def gladia_api_key(self) -> str:
        return os.getenv("GLADIA_API_KEY")

    @property
    def assemblyai_api_key(self) -> str:
        return os.getenv("ASSEMBLYAI_API_KEY")

    @property
    def rime_api_key(self) -> str:
        return os.getenv("RIME_API_KEY")

    @property
    def aws_access_key_id(self) -> str:
        return os.getenv("AWS_ACCESS_KEY_ID")

    @property
    def aws_secret_access_key(self) -> str:
        return os.getenv("AWS_SECRET_ACCESS_KEY")

    @property
    def aws_region(self) -> str:
        return os.getenv("AWS_REGION", "us-east-1")

    @property
    def ultravox_api_key(self) -> str:
        return os.getenv("ULTRAVOX_API_KEY")

    ###########################################################################
    # Bot configuration
    ###########################################################################

    @property
    def architecture_type(self) -> BotType:
        return self._bot_type

    @architecture_type.setter
    def architecture_type(self, value: BotType):
        self._bot_type = value
        os.environ["ARCHITECTURE_TYPE"] = value

    @property
    def bot_name(self) -> str:
        return os.getenv("BOT_NAME", "Marissa")

    @bot_name.setter
    def bot_name(self, value: str):
        os.environ["BOT_NAME"] = value

    ###########################################################################
    # LLM configuration
    ###########################################################################

    @property
    def llm_provider(self) -> str:
        return os.getenv("LLM_PROVIDER", "google").lower()

    @llm_provider.setter
    def llm_provider(self, value: str):
        os.environ["LLM_PROVIDER"] = value.lower()

    @property
    def llm_model(self) -> str:
        match self.llm_provider:
            case "google":
                return os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            case "openai":
                return os.getenv("OPENAI_MODEL", "gpt-4o")
            case "anthropic":
                return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
            case "groq":
                return os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
            case "together":
                return os.getenv("TOGETHER_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
            case "mistral":
                return os.getenv("MISTRAL_MODEL", "mistral-large-latest")
            case "ultravox":
                return os.getenv("ULTRAVOX_MODEL", "fixie-ai/ultravox")
            case _:
                return os.getenv("LLM_MODEL", "")

    @llm_model.setter
    def llm_model(self, value: str):
        match self.llm_provider:
            case "google":
                os.environ["GOOGLE_MODEL"] = value
            case "openai":
                os.environ["OPENAI_MODEL"] = value
            case "anthropic":
                os.environ["ANTHROPIC_MODEL"] = value
            case "groq":
                os.environ["GROQ_MODEL"] = value
            case "together":
                os.environ["TOGETHER_MODEL"] = value
            case "mistral":
                os.environ["MISTRAL_MODEL"] = value
            case "ultravox":
                os.environ["ULTRAVOX_MODEL"] = value

    @property
    def llm_temperature(self) -> float:
        return float(os.getenv("LLM_TEMPERATURE", 0.7))

    @property
    def llm_params(self) -> dict:
        return {"temperature": self.llm_temperature}

    # Backward compatibility properties (kept for original bots if needed)
    @property
    def google_model(self) -> str:
        return self.llm_model

    @property
    def google_params(self) -> GoogleLLMService.InputParams:
        return GoogleLLMService.InputParams(temperature=self.llm_params["temperature"])

    @property
    def openai_model(self) -> str:
        return self.llm_model

    @property
    def openai_params(self) -> BaseOpenAILLMService.InputParams:
        return BaseOpenAILLMService.InputParams(temperature=self.llm_params["temperature"])

    ###########################################################################
    # STT configuration
    ###########################################################################

    @property
    def stt_provider(self) -> str:
        provider = os.getenv("STT_PROVIDER")
        # If multimodal, default to the native provider unless an explicit non-default is provided
        if self.architecture_type == "multimodal":
             if not provider or provider.lower() in ["deepgram"]:
                  return self.llm_provider
        return (provider or "deepgram").lower()

    @stt_provider.setter
    def stt_provider(self, value: str):
        os.environ["STT_PROVIDER"] = value.lower()

    @property
    def stt_language(self) -> str:
        return os.getenv("STT_LANGUAGE", "en")

    @stt_language.setter
    def stt_language(self, value: str):
        os.environ["STT_LANGUAGE"] = value

    @property
    def stt_model(self) -> str:
        match self.stt_provider:
            case "deepgram":
                return os.getenv("DEEPGRAM_STT_MODEL", "nova-3-general")
            case _:
                return os.getenv("STT_MODEL", "")

    ###########################################################################
    # TTS configuration
    ###########################################################################

    @property
    def tts_provider(self) -> str:
        provider = os.getenv("TTS_PROVIDER")
        # If multimodal, default to the native provider unless an explicit non-default is provided
        if self.architecture_type == "multimodal":
             if not provider or provider.lower() in ["cartesia", "deepgram"]:
                  return self.llm_provider
        return (provider or "deepgram").lower()

    @tts_provider.setter
    def tts_provider(self, value: str):
        os.environ["TTS_PROVIDER"] = value.lower()

    @property
    def tts_language(self) -> str:
        return os.getenv("TTS_LANGUAGE", "en")

    @tts_language.setter
    def tts_language(self, value: str):
        os.environ["TTS_LANGUAGE"] = value

    @property
    def tts_speed(self) -> float:
        return float(os.getenv("TTS_SPEED", 1.0))

    @property
    def tts_voice(self) -> str:
        match self.tts_provider:
            case "deepgram":
                return os.getenv("DEEPGRAM_VOICE", "aura-athena-en")
            case "cartesia":
                return os.getenv("CARTESIA_VOICE", "79a125e8-cd45-4c13-8a67-188112f4dd22")
            case "elevenlabs":
                return os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
            case "playht":
                return os.getenv("PLAYHT_VOICE_ID", "s3://voice-training-authenticated/original_voices/marissa_extracted/manifest.json")
            case "rime":
                return os.getenv("RIME_VOICE_ID", "marissa")
            case "openai":
                return os.getenv("OPENAI_VOICE", "alloy")
            case "azure":
                return os.getenv("AZURE_VOICE", "en-US-AvaMultilingualNeural")
            case "ultravox":
                return os.getenv("ULTRAVOX_VOICE", "a6afd1fc-960f-45d3-9e46-e8182af650b9") # Default to 'Clive'
            case _:
                return os.getenv("TTS_VOICE", "")

    # Backward compatibility properties
    @property
    def deepgram_voice(self) -> str:
        return self.tts_voice if self.tts_provider == "deepgram" else os.getenv("DEEPGRAM_VOICE", "aura-athena-en")

    @property
    def cartesia_voice(self) -> str:
        return self.tts_voice if self.tts_provider == "cartesia" else os.getenv("CARTESIA_VOICE", "79a125e8-cd45-4c13-8a67-188112f4dd22")

    @property
    def elevenlabs_voice_id(self) -> str:
        return self.tts_voice if self.tts_provider == "elevenlabs" else os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

    @property
    def rime_voice_id(self) -> str:
        return self.tts_voice if self.tts_provider == "rime" else os.getenv("RIME_VOICE_ID", "marissa")

    @property
    def rime_reduce_latency(self) -> bool:
        return self._is_truthy(os.getenv("RIME_REDUCE_LATENCY", "false"))

    @property
    def rime_speed_alpha(self) -> float:
        return float(os.getenv("RIME_SPEED_ALPHA", 1.0))

    ###########################################################################
    # Filters and Extras
    ###########################################################################

    @property
    def enable_stt_mute_filter(self) -> bool:
        return self._is_truthy(os.getenv("ENABLE_STT_MUTE_FILTER", "false"))

    @property
    def amd_enabled(self) -> bool:
        return self._is_truthy(os.getenv("AMD_ENABLED", "false"))
        
    @property
    def speak_first(self) -> bool:
        return self._is_truthy(os.getenv("SPEAK_FIRST", "true"))

    @speak_first.setter
    def speak_first(self, value: bool):
        os.environ["SPEAK_FIRST"] = "true" if value else "false"

    @property
    def classifier_model(self) -> str:
        return os.getenv("CLASSIFIER_MODEL", "gemini-2.0-flash")
