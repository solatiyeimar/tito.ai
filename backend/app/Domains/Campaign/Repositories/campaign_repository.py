from abc import ABC, abstractmethod
from typing import List, Optional

from app.Domains.Campaign.Models.campaign import Campaign


class CampaignRepository(ABC):
    @abstractmethod
    def save(self, campaign: Campaign) -> None:
        pass

    @abstractmethod
    def get(self, campaign_id: str) -> Optional[Campaign]:
        pass

    @abstractmethod
    def list_all(self) -> List[Campaign]:
        pass

    @abstractmethod
    def delete(self, campaign_id: str) -> None:
        pass
