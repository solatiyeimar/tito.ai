import glob
import json
import os
from typing import List, Optional

from loguru import logger

from app.Domains.Prompt.Models.prompt import Prompt
from app.Domains.Prompt.Repositories.prompt_repository import PromptRepository


class FilePromptRepository(PromptRepository):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _get_file_path(self, prompt_id: str) -> str:
        return os.path.join(self.data_dir, f"{prompt_id}.json")

    def save(self, prompt: Prompt) -> Prompt:
        file_path = self._get_file_path(prompt.id)
        with open(file_path, "w") as f:
            json.dump(prompt.model_dump(mode="json"), f, indent=4)
        return prompt

    def get(self, prompt_id: str) -> Optional[Prompt]:
        file_path = self._get_file_path(prompt_id)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return Prompt(**data)
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_id}: {e}")
            return None

    def get_by_name(self, name: str) -> Optional[Prompt]:
        # This is inefficient for files, but acceptable for smaller sets. 
        # In a real DB, this would be a query.
        all_prompts = self.list_all()
        for prompt in all_prompts:
            if prompt.name == name:
                return prompt
        return None

    def list_all(self) -> List[Prompt]:
        prompts = []
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        for file_path in files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    prompts.append(Prompt(**data))
            except Exception:
                continue
        return prompts

    def delete(self, prompt_id: str) -> bool:
        file_path = self._get_file_path(prompt_id)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
