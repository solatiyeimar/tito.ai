import aiohttp
import json
import os
from loguru import logger

async def analyze_conversation_with_gemini(api_key: str, transcript: list) -> dict:
    """Helper to analyze conversation using Gemini API via REST."""
    if not api_key:
        return {"error": "No Google API Key available for analysis"}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    prompt = f"""
    Analyze the following conversation transcript and return a JSON object with:
    - sentiment (positive, neutral, negative)
    - summary (1-2 sentences)
    - user_intent (what did they want?)
    - resolution_status (resolved, unresolved, transferred)
    
    Transcript:
    {json.dumps(transcript)}
    
    Return ONLY raw JSON, no markdown blocks.
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Analysis failed: {text}")
                return {"error": "Analysis API call failed"}
            
            data = await resp.json()
            try:
                text_result = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text_result)
            except Exception as e:
                logger.error(f"Failed to parse analysis result: {e}")
                return {"error": "Failed to parse analysis"}
