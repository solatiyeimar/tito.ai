import os
from typing import Any, Dict, List, Optional

from loguru import logger
from pipecat.transcriptions.language import Language


class ServiceFactory:
    """Factory for creating Pipecat services (STT, TTS, LLM) based on configuration."""

    @staticmethod
    def create_stt_service(config):
        """Initialize the STT service based on configuration."""
        match config.stt_provider:
            case "deepgram":
                # Define a custom class to support detect_language and safe finalize (moved from base_bot.py)
                # We need to ensure we don't redefine it if it's not needed, or keep it local
                from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
                from pipecat.services.deepgram.stt import DeepgramSTTService

                # Re-implementing the custom logic locally within the factory method
                # Ideally, this custom class should be in a separate file if it grows,
                # but for now, keeping it close to where it's used is fine for the factory.

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
                                    logger.trace(
                                        f"Triggered finalize event on: {frame.name=}, {direction=}"
                                    )
                            except Exception:
                                pass

                keywords = []
                stt_keywords_env = os.getenv("STT_KEYWORDS")
                if stt_keywords_env:
                    keywords = stt_keywords_env.split(",")

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
                    api_key=config.deepgram_api_key, live_options=CustomLiveOptions(**options)
                )
            case _:
                raise ValueError(f"Invalid STT provider: {config.stt_provider}")

    @staticmethod
    def create_tts_service(config):
        """Initialize the TTS service based on configuration."""
        match config.tts_provider:
            case "cartesia":
                from pipecat.services.cartesia.tts import CartesiaTTSService

                return CartesiaTTSService(
                    api_key=config.cartesia_api_key,
                    voice_id=config.tts_voice,
                    params=CartesiaTTSService.InputParams(
                        language=Language(config.tts_language) if config.tts_language else Language.EN,
                    ),
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

    @staticmethod
    def create_llm_service(config, system_messages: List[Dict[str, str]]):
        """Initialize the LLM service based on configuration."""
        if system_messages:
            system_instruction = system_messages[0]["content"]
        else:
            from app.Domains.Agent.Prompts.helpers import get_prompt_service
            service = get_prompt_service()
            system_instruction = service.render_prompt("default.system_prompt")
            if not system_instruction:
                system_instruction = "You are a voice assistant"

        match config.llm_provider:
            case "google":
                from pipecat.services.google.llm import GoogleLLMService

                return GoogleLLMService(
                    api_key=config.google_api_key,
                    model=config.llm_model,
                    params=config.google_params,
                    system_instruction=system_instruction,
                    tools=config.tools,
                )
            case "openai":
                from pipecat.services.openai.llm import OpenAILLMService

                return OpenAILLMService(
                    api_key=config.openai_api_key,
                    model=config.llm_model,
                    params=config.openai_params,
                )
            case "anthropic":
                from pipecat.services.anthropic.llm import AnthropicLLMService

                return AnthropicLLMService(api_key=config.anthropic_api_key, model=config.llm_model)
            case "groq":
                from pipecat.services.groq.llm import GroqLLMService

                return GroqLLMService(api_key=config.groq_api_key, model=config.llm_model)
            case "together":
                from pipecat.services.together import TogetherLLMService

                return TogetherLLMService(api_key=config.together_api_key, model=config.llm_model)
            case "mistral":
                from pipecat.services.mistral import MistralLLMService

                return MistralLLMService(api_key=config.mistral_api_key, model=config.llm_model)
            case "ultravox":
                from pipecat.services.ultravox.llm import (
                    OneShotInputParams,
                    UltravoxRealtimeLLMService,
                )

                return UltravoxRealtimeLLMService(
                    params=OneShotInputParams(
                        api_key=config.ultravox_api_key,
                        model=config.llm_model,
                        voice=config.tts_voice,
                        system_prompt=system_instruction,
                        temperature=config.llm_temperature or 0.3,
                    )
                )
            case _:
                raise ValueError(f"Invalid LLM provider: {config.llm_provider}")
