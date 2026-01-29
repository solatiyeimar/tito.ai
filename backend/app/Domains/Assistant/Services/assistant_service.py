import os
from typing import Dict, List, Optional

from app.Core.Config.bot import BotConfig  # We'll need a way to mock/create config on the fly
from app.Domains.Agent.Factory.service_factory import ServiceFactory
from app.Domains.Assistant.Models.assistant import Assistant
from app.Domains.Assistant.Repositories.assistant_repository import AssistantRepository


class AssistantService:
    def __init__(self, repository: AssistantRepository):
        self.repository = repository

    def create_assistant(self, assistant: Assistant) -> Assistant:
        return self.repository.save(assistant)

    def get_assistant(self, assistant_id: str) -> Optional[Assistant]:
        return self.repository.get(assistant_id)

    def list_assistants(self) -> List[Assistant]:
        return self.repository.list_all()

    def update_assistant(self, assistant_id: str, updates: Dict) -> Optional[Assistant]:
        assistant = self.repository.get(assistant_id)
        if not assistant:
            return None

        # Pydantic v2 update logic (deep merge preferred, but simple dict update here)
        updated_data = assistant.model_dump()

        # Helper for recursive update could be added here
        # For now assuming full object replacement or top-level keys
        updated_data.update(updates)

        new_assistant = Assistant(**updated_data)
        new_assistant.id = assistant_id  # Ensure ID preservation
        return self.repository.save(new_assistant)

    def delete_assistant(self, assistant_id: str) -> bool:
        return self.repository.delete(assistant_id)

    async def chat_with_assistant(self, assistant_id: str, message: str) -> str:
        """
        Interacts with the assistant's LLM directly (text-only) for testing.
        """
        assistant = self.repository.get(assistant_id)
        if not assistant:
            raise ValueError("Assistant not found")

        # Create a temporary BotConfig
        # We need to map Assistant config to BotConfig environment variables or object
        # Since ServiceFactory reads from Config object, we construct one manually

        class MockConfig:
            def __init__(self, assistant: Assistant):
                from pipecat.services.google.llm import GoogleLLMService
                from pipecat.services.openai.llm import BaseOpenAILLMService

                self.llm_provider = assistant.agent.provider
                self.llm_model = assistant.agent.model
                self.llm_temperature = assistant.agent.temperature
                self.google_api_key = os.getenv("GOOGLE_API_KEY")
                self.openai_api_key = os.getenv("OPENAI_API_KEY")
                self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                self.groq_api_key = os.getenv("GROQ_API_KEY")
                # Add other keys as needed

                # Fix: Use proper InputParams objects instead of dicts
                self.google_params = GoogleLLMService.InputParams(temperature=self.llm_temperature)
                self.openai_params = BaseOpenAILLMService.InputParams(
                    temperature=self.llm_temperature
                )
                self.tools = assistant.agent.tools

        config = MockConfig(assistant)

        # Create LLM service
        system_messages = [{"role": "system", "content": assistant.agent.system_prompt}]
        llm_service = ServiceFactory.create_llm_service(config, system_messages)

        # Ideally LLMService in Pipecat works with frames, but many expose .chat_completion or similar?
        # Standard Pipecat LLM services process *frames*.
        # To just "chat" directly, we might need to use the underlying SDK or a specific method if exposed.
        # Pipecat services don't typically expose a direct "chat(text) -> text" method easily without the pipeline.

        # ALTERNATIVE: Use the underlying provider SDKs directly based on config.
        # Or check if Pipecat services have a helper.

        # Let's try to use the LLMService instance if it supports standard completion,
        # otherwise, fall back to a simple direct call for "Chat API".

        # For now, implementing a simple direct call using standard libraries for common providers
        # to ensure reliability of this "Chat API" endpoint.

        if assistant.agent.provider == "openai":
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = await client.chat.completions.create(
                model=assistant.agent.model or "gpt-5o",
                messages=[
                    {"role": "system", "content": assistant.agent.system_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=assistant.agent.temperature,
            )
            return response.choices[0].message.content

        elif assistant.agent.provider == "google":
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(
                model_name=assistant.agent.model or "gemini-2.5-flash",
                system_instruction=assistant.agent.system_prompt,
            )
            response = await model.generate_content_async(message)
            return response.text

        # Add other providers...
        return "Chat provider not supported for text-only test yet."
