import os

GET_SECURE_DATA_TOOL = {
    "type": "function",
    "function": {
        "name": "get_secure_data",
        "description": "Retrieves secure data using the injected secret token.",
        "parameters": {
            "type": "object",
            "properties": {},
        }
    }
}

async def get_secure_data_handler(function_name, tool_call_id, args, llm, context, result_callback):
    token = os.environ.get("CRM_SECRET_TOKEN")
    if token == "super-secret-123":
        await result_callback({"data": "Access Granted: VIP Customer List [Alice, Bob]"})
    else:
        await result_callback({"error": f"Access Denied. Token was: {token}"})
