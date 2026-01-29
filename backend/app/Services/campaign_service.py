import asyncio
import os
import json
from typing import List, Optional, Dict, Any
from loguru import logger
from app.Http.DTOs.campaign_schemas import CampaignConfig, Contact
from app.Services.runner_service import start_bot_process, create_room_and_token
from app.Services.assistant_store import AssistantStore
from app.Core.Parsers.bot_config_parser import dict_to_cli_args

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "resources", "data", "campaigns")

class CampaignManager:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.active_campaigns: Dict[str, asyncio.Task] = {}

    def _get_file_path(self, campaign_id: str) -> str:
        return os.path.join(self.data_dir, f"{campaign_id}.json")

    def save_campaign(self, campaign: CampaignConfig):
        file_path = self._get_file_path(campaign.id)
        with open(file_path, "w") as f:
            json.dump(campaign.model_dump(mode='json'), f, indent=4)

    def get_campaign(self, campaign_id: str) -> Optional[CampaignConfig]:
        file_path = self._get_file_path(campaign_id)
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r") as f:
            data = json.load(f)
            return CampaignConfig(**data)

    async def start_campaign(self, campaign_id: str):
        if campaign_id in self.active_campaigns:
            logger.warning(f"Campaign {campaign_id} is already running")
            return

        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return

        campaign.status = "active"
        self.save_campaign(campaign)
        
        task = asyncio.create_task(self._run_outbound_loop(campaign_id))
        self.active_campaigns[campaign_id] = task
        logger.info(f"🚀 Started proactive campaign: {campaign.name}")

    async def _run_outbound_loop(self, campaign_id: str):
        """Worker loop to process contacts in parallel based on concurrency."""
        assistant_store = AssistantStore()
        
        while True:
            campaign = self.get_campaign(campaign_id)
            if not campaign or campaign.status != "active":
                break
                
            pending_contacts = [c for c in campaign.contacts if c.status == "pending"]
            if not pending_contacts:
                campaign.status = "completed"
                self.save_campaign(campaign)
                logger.info(f"✅ Campaign {campaign.name} completed.")
                break

            # Respect concurrency
            to_process = pending_contacts[:campaign.concurrency]
            
            tasks = []
            for contact in to_process:
                tasks.append(self._dial_contact(campaign, contact, assistant_store))
            
            await asyncio.gather(*tasks)
            # Sleep a bit between chunks to avoid flooding
            await asyncio.sleep(2)

    async def _dial_contact(self, campaign: CampaignConfig, contact: Contact, assistant_store: AssistantStore):
        logger.info(f"☎️ Dialing {contact.name} ({contact.phone}) for campaign {campaign.name}")
        
        assistant = assistant_store.get_assistant(campaign.assistant_id)
        if not assistant:
             contact.status = "failed"
             self.save_campaign(campaign)
             return

        # Prepare arguments
        config_args = dict_to_cli_args(assistant)
        config_args.extend(["--assistant-id", campaign.assistant_id])
        
        # Merge campaign variables with contact variables
        vars = {**contact.variables, "campaign_name": campaign.name}
        config_args.extend(["--prompt-variables", json.dumps(vars)])
        
        try:
            room_url, token = await create_room_and_token()
            # In a real outbound scenario, we would dial the phone number here
            # via Daily SIP or another provider. For now we simulate.
            
            pid = await start_bot_process(room_url, token, extra_args=config_args)
            
            # Update contact status
            for c in campaign.contacts:
                if c.id == contact.id:
                    c.status = "called"
                    c.last_call_id = str(pid)
            
            self.save_campaign(campaign)
            logger.success(f"Bot started for {contact.phone} (PID: {pid})")
            
        except Exception as e:
            logger.error(f"Failed to dial {contact.phone}: {e}")
            for c in campaign.contacts:
                if c.id == contact.id:
                    c.status = "failed"
            self.save_campaign(campaign)
