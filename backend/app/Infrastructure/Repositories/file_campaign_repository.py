import json
import os
from typing import List, Optional

from app.Domains.Campaign.Models.campaign import Campaign
from app.Domains.Campaign.Repositories.campaign_repository import CampaignRepository


class FileCampaignRepository(CampaignRepository):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _get_file_path(self, campaign_id: str) -> str:
        return os.path.join(self.data_dir, f"{campaign_id}.json")

    def save(self, campaign: Campaign) -> None:
        file_path = self._get_file_path(campaign.id)
        with open(file_path, "w") as f:
            # model_dump is Pydantic v2, ensure compatibility
            json.dump(campaign.dict(), f, indent=4, default=str)

    def get(self, campaign_id: str) -> Optional[Campaign]:
        file_path = self._get_file_path(campaign_id)
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r") as f:
            data = json.load(f)
            return Campaign(**data)

    def list_all(self) -> List[Campaign]:
        campaigns = []
        if os.path.exists(self.data_dir):
            for f in os.listdir(self.data_dir):
                if f.endswith(".json"):
                    with open(os.path.join(self.data_dir, f), "r") as file:
                        try:
                            campaigns.append(Campaign(**json.load(file)))
                        except Exception:
                            continue  # Skip malformed files
        return campaigns

    def delete(self, campaign_id: str) -> None:
        file_path = self._get_file_path(campaign_id)
        if os.path.exists(file_path):
            os.remove(file_path)
