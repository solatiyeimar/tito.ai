import asyncio
import os

from fastapi import Request

from app.Domains.Assistant.Services.assistant_service import AssistantService
from app.Domains.Call.Services.call_service import CallService
from app.Domains.Campaign.Services.campaign_service import CampaignService
from app.Infrastructure.Call.daily_room_provider import DailyRoomProvider
from app.Infrastructure.Call.local_bot_process_manager import LocalBotProcessManager
from app.Infrastructure.Repositories.file_assistant_repository import FileAssistantRepository
from app.Infrastructure.Repositories.file_campaign_repository import FileCampaignRepository

# Shared data directories
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSISTANT_DATA_DIR = os.path.join(
    BACKEND_ROOT,
    "resources",
    "data",
    "assistants",
)

CAMPAIGN_DATA_DIR = os.path.join(
    BACKEND_ROOT,
    "resources",
    "data",
    "campaigns",
)

# Singletons (Infrastructure)
_room_provider = DailyRoomProvider()
_process_manager = LocalBotProcessManager(_room_provider)


# Helper to start cleanup task
async def start_process_cleanup():
    await _process_manager.cleanup()


def get_room_provider() -> DailyRoomProvider:
    return _room_provider


def get_process_manager() -> LocalBotProcessManager:
    return _process_manager


def get_assistant_service() -> AssistantService:
    repo = FileAssistantRepository(ASSISTANT_DATA_DIR)
    return AssistantService(repo)


def get_campaign_service() -> CampaignService:
    repo = FileCampaignRepository(CAMPAIGN_DATA_DIR)
    return CampaignService(repo, _room_provider, _process_manager)


def get_call_service() -> CallService:
    # We construct CallService on demand, but injecting the singleton infra components
    assistant_service = get_assistant_service()
    return CallService(assistant_service, _room_provider, _process_manager)
