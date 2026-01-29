import asyncio
import json
import os

from loguru import logger

from app.Core.Parsers.bot_config_parser import dict_to_cli_args
from app.Domains.Assistant.Services.assistant_service import AssistantService
from app.Domains.Call.Interfaces.bot_process_manager import BotProcessManager
from app.Domains.Call.Interfaces.room_provider import RoomProvider
from app.Domains.Campaign.Models.campaign import Campaign
from app.Domains.Campaign.Repositories.campaign_repository import CampaignRepository
from app.Infrastructure.Repositories.file_assistant_repository import FileAssistantRepository


class CampaignService:
    def __init__(
        self,
        repository: CampaignRepository,
        room_provider: RoomProvider,
        process_manager: BotProcessManager,
    ):
        self.repository = repository
        self.room_provider = room_provider
        self.process_manager = process_manager
        self.active_tasks = {}  # In-memory task tracking (simple version)

        # Setup assistant service for dialing
        assistant_data_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ),
            "resources",
            "data",
            "assistants",
        )
        assistant_repo = FileAssistantRepository(assistant_data_dir)
        self.assistant_service = AssistantService(assistant_repo)

    def create_campaign(self, campaign_data: Campaign) -> Campaign:
        self.repository.save(campaign_data)
        return campaign_data

    def get_campaign(self, campaign_id: str) -> Campaign:
        return self.repository.get(campaign_id)

    def list_campaigns(self) -> list[Campaign]:
        return self.repository.list_all()

    async def start_campaign_background(self, campaign_id: str):
        if campaign_id in self.active_tasks:
            logger.warning(f"Campaign {campaign_id} is already running")
            return

        campaign = self.repository.get(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")

        campaign.status = "active"
        self.repository.save(campaign)

        # Start background worker
        task = asyncio.create_task(self._run_outbound_loop(campaign.id))
        self.active_tasks[campaign_id] = task
        logger.info(f"🚀 Started proactive campaign: {campaign.name}")

    async def _run_outbound_loop(self, campaign_id: str):
        while True:
            # Re-fetch campaign to get latest status/contacts
            campaign = self.repository.get(campaign_id)
            if not campaign or campaign.status != "active":
                break

            pending_contacts = [c for c in campaign.contacts if c.status == "pending"]
            if not pending_contacts:
                campaign.status = "completed"
                self.repository.save(campaign)
                logger.info(f"✅ Campaign {campaign.name} completed.")
                break

            to_process = pending_contacts[: campaign.concurrency]
            tasks = []
            for contact in to_process:
                tasks.append(self._dial_contact(campaign, contact))

            if tasks:
                await asyncio.gather(*tasks)

            await asyncio.sleep(2)

        # Cleanup
        if campaign_id in self.active_tasks:
            del self.active_tasks[campaign_id]

    async def _dial_contact(self, campaign: Campaign, contact):
        # ... logic similar to original ...
        logger.info(f"☎️ Dialing {contact.name} ({contact.phone})")

        assistant = self.assistant_service.get_assistant(campaign.assistant_id)
        if not assistant:
            contact.status = "failed"
            self.repository.save(campaign)
            return

        config_args = dict_to_cli_args(assistant)
        config_args.extend(["--assistant-id", campaign.assistant_id])
        config_args.extend(["--agent-type", "outbound"])  # Use new agent type logic

        vars = {**contact.variables, "campaign_name": campaign.name}
        config_args.extend(["--prompt-variables", json.dumps(vars)])

        try:
            room_url, token = await self.room_provider.create_room_and_token()
            pid = await self.process_manager.start_bot(room_url, token, args=config_args)

            # Update contact in memory then save
            contact.status = "called"
            contact.last_call_id = str(pid)
            self.repository.save(campaign)

        except Exception as e:
            logger.error(f"Failed to dial: {e}")
            contact.status = "failed"
            self.repository.save(campaign)
