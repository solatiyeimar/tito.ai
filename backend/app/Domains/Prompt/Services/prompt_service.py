from typing import Any, Dict, List, Optional
from jinja2 import Template

from app.Domains.Prompt.Models.prompt import Prompt
from app.Domains.Prompt.Repositories.prompt_repository import PromptRepository


class PromptService:
    def __init__(self, repository: PromptRepository):
        self.repository = repository

    def create_prompt(self, prompt: Prompt) -> Prompt:
        return self.repository.save(prompt)

    def get_prompt(self, prompt_id: str) -> Optional[Prompt]:
        return self.repository.get(prompt_id)

    def get_prompt_by_name(self, name: str) -> Optional[Prompt]:
        return self.repository.get_by_name(name)

    def list_prompts(self) -> List[Prompt]:
        return self.repository.list_all()

    def update_prompt(self, prompt_id: str, updates: Dict) -> Optional[Prompt]:
        prompt = self.repository.get(prompt_id)
        if not prompt:
            return None

        updated_data = prompt.model_dump()
        updated_data.update(updates)
        
        # Ensure ID doesn't change
        updated_data["id"] = prompt_id
        
        new_prompt = Prompt(**updated_data)
        return self.repository.save(new_prompt)

    def delete_prompt(self, prompt_id: str) -> bool:
        return self.repository.delete(prompt_id)

    def render_prompt(self, name: str, context: Dict[str, Any] = None) -> str:
        """
        Fetches a prompt by name and renders it with the given context.
        Falls back to returning the name or empty string if not found (or raises error depending on preference).
        """
        prompt = self.get_prompt_by_name(name)
        if not prompt:
            raise ValueError(f"Prompt with name '{name}' not found.")
        
        context = context or {}
        
        # Use Jinja2 for rendering
        template = Template(prompt.template)
        return template.render(**context)

    def render_raw_template(self, template_str: str, context: Dict[str, Any] = None) -> str:
        """
        Helper to render a raw template string without saving it first.
        Useful for testing or dynamic construction.
        """
        context = context or {}
        template = Template(template_str)
        return template.render(**context)
