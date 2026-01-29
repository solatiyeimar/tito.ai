from app.Core.Config.bot import BotConfig

from .helpers import get_current_date_uk, get_system_prompt, get_prompt_service
from .types import NodeContent

config = BotConfig()


def get_simple_prompt() -> NodeContent:
    """Return a dictionary with the simple prompt in Spanish, combining all flows."""
    service = get_prompt_service()
    prompt_text = service.render_prompt(
        "agent.simple.system_prompt",
        {"bot_name": config.bot_name, "current_date": get_current_date_uk()},
    )
    return get_system_prompt(prompt_text)
