
import os
import json
import glob
from typing import List, Optional
from loguru import logger
from app.Http.DTOs.schemas import AssistantConfig

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "resources", "data", "assistants")

class AssistantStore:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _resolve_id(self, assistant_id: str) -> str:
        mapping_path = os.path.join(self.data_dir, "migration_mapping.json")
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, "r") as f:
                    mapping = json.load(f)
                    return mapping.get(assistant_id, assistant_id)
            except Exception as e:
                logger.error(f"Error reading migration mapping: {e}")
        return assistant_id

    def _get_file_path(self, assistant_id: str) -> str:
        actual_id = self._resolve_id(assistant_id)
        return os.path.join(self.data_dir, f"{actual_id}.json")

    def list_assistants(self) -> List[AssistantConfig]:
        assistants = []
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        for file_path in files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    assistants.append(AssistantConfig(**data))
            except Exception as e:
                logger.error(f"Error loading assistant from {file_path}: {e}")
        return assistants

    def get_assistant(self, assistant_id: str) -> Optional[AssistantConfig]:
        file_path = self._get_file_path(assistant_id)
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return AssistantConfig(**data)
        except Exception as e:
            logger.error(f"Error loading assistant {assistant_id}: {e}")
            return None

    def create_assistant(self, assistant: AssistantConfig) -> AssistantConfig:
        file_path = self._get_file_path(assistant.id)
        if os.path.exists(file_path):
             raise ValueError(f"Assistant with ID {assistant.id} already exists")
             
        with open(file_path, "w") as f:
            json.dump(assistant.model_dump(), f, indent=4)
        return assistant

    def update_assistant(self, assistant_id: str, assistant: AssistantConfig) -> Optional[AssistantConfig]:
        file_path = self._get_file_path(assistant_id)
        if not os.path.exists(file_path):
            return None
            
        # Ensure ID consistency
        assistant.id = assistant_id
        
        with open(file_path, "w") as f:
            json.dump(assistant.model_dump(), f, indent=4)
        return assistant

    def delete_assistant(self, assistant_id: str) -> bool:
        file_path = self._get_file_path(assistant_id)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
