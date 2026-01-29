from datetime import datetime
from typing import List, Optional
from app.Infrastructure.Repositories.json_crm_repository import JsonCRMRepository
from app.Domains.CRM.Models.schemas import Lead, Appointment

class CRMService:
    def __init__(self, repository: JsonCRMRepository):
        self.repository = repository

    async def create_lead(self, name: str, email: str, phone: Optional[str] = None) -> Lead:
        existing_lead = await self.repository.get_lead_by_email(email)
        if existing_lead:
            return existing_lead
        
        lead = Lead(name=name, email=email, phone=phone)
        return await self.repository.add_lead(lead)

    async def get_lead(self, email: str) -> Optional[Lead]:
        return await self.repository.get_lead_by_email(email)

    async def schedule_appointment(self, lead_id: str, date_time: str) -> Appointment:
        # Convert user string to datetime if needed, assumes ISO 8601 for now
        dt = datetime.fromisoformat(date_time.replace("Z", "+00:00"))
        appointment = Appointment(lead_id=lead_id, datetime=dt)
        return await self.repository.add_appointment(appointment)

    async def get_appointments(self, lead_id: str) -> List[Appointment]:
        return await self.repository.get_appointments_by_lead(lead_id)
