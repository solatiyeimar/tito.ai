from datetime import datetime

import pytz

from .types import NodeMessage
import os
from app.Domains.Prompt.Services.prompt_service import PromptService
from app.Infrastructure.Repositories.file_prompt_repository import FilePromptRepository

def get_prompt_service() -> PromptService:
    # Calculate path to backend/resources/data/prompts
    # Current file: .../backend/app/Domains/Agent/Prompts/helpers.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 5 levels to backend/
    backend_dir = os.path.abspath(os.path.join(current_dir, "../../../../../"))
    prompts_dir = os.path.join(backend_dir, "resources", "data", "prompts")
    
    repository = FilePromptRepository(prompts_dir)
    return PromptService(repository)


def get_system_prompt(content: str) -> NodeMessage:
    """Return a dictionary with a system prompt."""
    return {
        "role_messages": [],
        "task_messages": [
            {
                "role": "system",
                "content": content,
            }
        ],
    }


def get_current_date_uk() -> str:
    """Return the current day and date formatted for the UK timezone."""
    current_date = datetime.now(pytz.timezone("America/Bogota"))
    return current_date.strftime("%A, %d %B %Y")
