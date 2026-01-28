from fastapi import FastAPI, Request
import uvicorn
import json

app = FastAPI()

@app.post("/webhook")
async def receive_webhook(request: Request):
    try:
        data = await request.json()
        print("\n" + "="*40)
        print(f"🔔 WEBHOOK RECEIVED [{data.get('event', 'unknown')}]")
        print("-" * 40)
        print(json.dumps(data, indent=2))
        print("="*40 + "\n")
    except Exception as e:
        print(f"Error processing webhook: {e}")
    return {"status": "received"}

if __name__ == "__main__":
    print("🚀 Webhook Receiver running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
