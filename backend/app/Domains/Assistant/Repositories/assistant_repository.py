from abc import ABC, abstractmethod
from typing import List, Optional

from app.Domains.Assistant.Models.assistant import Assistant


class AssistantRepository(ABC):
    @abstractmethod
    def save(self, assistant: Assistant) -> Assistant:
        pass

    @abstractmethod
    def get(self, assistant_id: str) -> Optional[Assistant]:
        pass

    @abstractmethod
    def list_all(self) -> List[Assistant]:
        pass

    @abstractmethod
    def delete(self, assistant_id: str) -> bool:
        pass
