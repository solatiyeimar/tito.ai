from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from app.Http.DTOs.campaign_schemas import CampaignConfig, CampaignResponse
from app.Http.DTOs.error_schemas import APIErrorResponse
from app.Services.campaign_service import CampaignManager

router = APIRouter(tags=["Campaigns"])
manager = CampaignManager()

@router.post(
    "/campaigns", 
    response_model=CampaignResponse, 
    status_code=201,
    summary="Create a new campaign",
    description="Configures a new outbound campaign.",
    responses={422: {"model": APIErrorResponse}}
)
async def create_campaign(campaign: CampaignConfig):
    """
    Saves a new campaign configuration to the database.
    """
    manager.save_campaign(campaign)
    return campaign

@router.get(
    "/campaigns", 
    response_model=List[CampaignResponse],
    summary="List all campaigns",
    description="Retrieve a list of all currently configured campaigns."
)
async def list_campaigns():
    """
    Returns a list of all campaigns stored in the data directory.
    """
    # Simple implementation: list files in data dir
    import os, json
    campaigns = []
    if os.path.exists(manager.data_dir):
        for f in os.listdir(manager.data_dir):
            if f.endswith(".json"):
                with open(os.path.join(manager.data_dir, f), "r") as file:
                    campaigns.append(CampaignConfig(**json.load(file)))
    return campaigns

@router.post(
    "/campaigns/{campaign_id}/start",
    summary="Start a campaign",
    description="Trigger the background process for an outbound campaign.",
    responses={404: {"model": APIErrorResponse}}
)
async def start_campaign(campaign_id: str, background_tasks: BackgroundTasks):
    """
    Locates a campaign by ID and starts its execution in the background.
    """
    campaign = manager.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    background_tasks.add_task(manager.start_campaign, campaign_id)
    return {"message": f"Campaign {campaign_id} started in background"}

@router.get(
    "/campaigns/{campaign_id}", 
    response_model=CampaignResponse,
    summary="Get campaign details",
    responses={404: {"model": APIErrorResponse}}
)
async def get_campaign(campaign_id: str):
    """
    Fetch a single campaign's configuration by its ID.
    """
    campaign = manager.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign
