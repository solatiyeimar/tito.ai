import glob
import json
import os
from typing import List, Optional

from loguru import logger

from app.Domains.Assistant.Models.assistant import Assistant
from app.Domains.Assistant.Repositories.assistant_repository import AssistantRepository


class FileAssistantRepository(AssistantRepository):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _resolve_id(self, assistant_id: str) -> str:
        mapping_path = os.path.join(self.data_dir, "migration_mapping.json")
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, "r") as f:
                    mapping = json.load(f)
                    return mapping.get(assistant_id, assistant_id)
            except Exception:
                pass
        return assistant_id

    def _get_file_path(self, assistant_id: str) -> str:
        actual_id = self._resolve_id(assistant_id)
        return os.path.join(self.data_dir, f"{actual_id}.json")

    def save(self, assistant: Assistant) -> Assistant:
        file_path = self._get_file_path(assistant.id)
        with open(file_path, "w") as f:
            json.dump(assistant.model_dump(mode="json"), f, indent=4)
        return assistant

    def get(self, assistant_id: str) -> Optional[Assistant]:
        file_path = self._get_file_path(assistant_id)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return Assistant(**data)
        except Exception as e:
            logger.error(f"Error loading assistant {assistant_id}: {e}")
            return None

    def list_all(self) -> List[Assistant]:
        assistants = []
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        for file_path in files:
            # Skip non-assistant files
            if "migration_mapping" in file_path or "campaign" in file_path:
                continue
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    # Simple validation check
                    if "agent" in data:
                        assistants.append(Assistant(**data))
            except Exception:
                continue
        return assistants

    def delete(self, assistant_id: str) -> bool:
        file_path = self._get_file_path(assistant_id)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
