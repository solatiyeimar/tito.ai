import json

from loguru import logger

from app.Core.Parsers.bot_config_parser import dict_to_cli_args
from app.Domains.Assistant.Services.assistant_service import AssistantService
from app.Domains.Call.Interfaces.bot_process_manager import BotProcessManager
from app.Domains.Call.Interfaces.room_provider import RoomProvider
from app.Domains.Call.Models.call import CallConfig, CallSession


class CallService:
    def __init__(
        self,
        assistant_service: AssistantService,
        room_provider: RoomProvider,
        process_manager: BotProcessManager,
    ):
        self.assistant_service = assistant_service
        self.room_provider = room_provider
        self.process_manager = process_manager

    async def initiate_call(self, config: CallConfig) -> CallSession:
        logger.info(f"Initiating call for assistant {config.assistant_id}")

        # 1. Validate Assistant
        assistant = self.assistant_service.get_assistant(config.assistant_id)
        if not assistant:
            raise ValueError("Assistant not found")

        # 2. Create Infrastructure (Room)
        room_url, token = await self.room_provider.create_room_and_token()

        # 3. Prepare Arguments
        cli_args = dict_to_cli_args(assistant)
        cli_args.extend(["--assistant-id", config.assistant_id])

        if config.variables:
            cli_args.extend(["--prompt-variables", json.dumps(config.variables)])

        if config.dynamic_vocabulary:
            keywords = ",".join(config.dynamic_vocabulary)
            cli_args.extend(["--stt-keywords", keywords])

        # 4. Start Process
        pid = await self.process_manager.start_bot(
            room_url=room_url, token=token, args=cli_args, env_vars=config.secrets
        )

        return CallSession(id=str(pid), room_url=room_url, token=token, status="initiated")

    def get_call_status(self, call_id: str) -> str:
        # Assuming call_id is PID for now
        try:
            return self.process_manager.get_status(int(call_id))
        except ValueError:
            return "unknown"

    async def start_rtvi_session(self, raw_config: dict) -> CallSession:
        """Starts a session with inline configuration (no saved assistant needed)"""
        room_url, token = await self.room_provider.create_room_and_token()

        cli_args = []
        arg_map = {
            "bot_type": "-a",
            "bot_name": "-n",
            "llm_provider": "-l",
            "llm_model": "-m",
            "llm_temperature": "-T",
            "stt_provider": "-s",
            "tts_provider": "-p",
            "tts_voice": "-v",
        }
        for key, flag in arg_map.items():
            if key in raw_config:
                cli_args.extend([flag, str(raw_config[key])])

        if "enable_stt_mute_filter" in raw_config:
            val = "true" if raw_config["enable_stt_mute_filter"] else "false"
            cli_args.extend(["--enable-stt-mute-filter", val])

        pid = await self.process_manager.start_bot(room_url, token, cli_args)

        return CallSession(id=str(pid), room_url=room_url, token=token, status="initiated")
