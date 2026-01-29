from typing import Any, Dict

TRANSFER_CALL_TOOL = {
    "type": "function",
    "function": {
        "name": "transfer_call",
        "description": "Transfers the current call to a human agent or another phone number.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "The phone number or SIP URI to transfer to. If not provided, uses default."
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the transfer (e.g., 'user requested human', 'too complex')."
                }
            },
            "required": ["reason"]
        }
    }
}
