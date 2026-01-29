from abc import ABC, abstractmethod
from typing import List, Optional

from app.Domains.Prompt.Models.prompt import Prompt


class PromptRepository(ABC):
    @abstractmethod
    def save(self, prompt: Prompt) -> Prompt:
        pass

    @abstractmethod
    def get(self, prompt_id: str) -> Optional[Prompt]:
        pass
    
    @abstractmethod
    def get_by_name(self, name: str) -> Optional[Prompt]:
        pass

    @abstractmethod
    def list_all(self) -> List[Prompt]:
        pass

    @abstractmethod
    def delete(self, prompt_id: str) -> bool:
        pass
