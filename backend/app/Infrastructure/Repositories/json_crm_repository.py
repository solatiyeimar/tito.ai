import json
import os
import aiofiles
from typing import List, Optional
from datetime import datetime

from app.Domains.CRM.Models.schemas import Lead, Appointment

class JsonCRMRepository:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.leads: List[Lead] = []
        self.appointments: List[Appointment] = []
        self._ensure_file_exists()
        self._load_data()

    def _ensure_file_exists(self):
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        if not os.path.exists(self.data_path):
            self._save_data_sync()

    def _save_data_sync(self):
        data = {
            "leads": [lead.model_dump(mode="json") for lead in self.leads],
            "appointments": [appt.model_dump(mode="json") for appt in self.appointments]
        }
        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    async def _save_data(self):
        data = {
            "leads": [lead.model_dump(mode="json") for lead in self.leads],
            "appointments": [appt.model_dump(mode="json") for appt in self.appointments]
        }
        async with aiofiles.open(self.data_path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    def _load_data(self):
        if not os.path.exists(self.data_path):
            return
        
        try:
            with open(self.data_path, "r") as f:
                data = json.load(f)
                self.leads = [Lead(**l) for l in data.get("leads", [])]
                self.appointments = [Appointment(**a) for a in data.get("appointments", [])]
        except json.JSONDecodeError:
            self.leads = []
            self.appointments = []

    async def add_lead(self, lead: Lead) -> Lead:
        self.leads.append(lead)
        await self._save_data()
        return lead

    async def get_lead_by_email(self, email: str) -> Optional[Lead]:
        for lead in self.leads:
            if lead.email == email:
                return lead
        return None

    async def add_appointment(self, appointment: Appointment) -> Appointment:
        self.appointments.append(appointment)
        await self._save_data()
        return appointment

    async def get_appointments_by_lead(self, lead_id: str) -> List[Appointment]:
        return [appt for appt in self.appointments if appt.lead_id == lead_id]

    async def get_all_appointments(self) -> List[Appointment]:
        return self.appointments
