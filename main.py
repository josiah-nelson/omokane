import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Get the API URL from the environment variable
LM_STUDIO_URL = os.getenv("LM_STUDIO_API_URL")

class PromptRequest(BaseModel):
    prompt: str
    # Add any other parameters you might want to pass, e.g., temperature
    temperature: float = 0.7

@app.post("/v1/process")
def process_prompt(request: PromptRequest):
    if not LM_STUDIO_URL:
        raise HTTPException(status_code=500, detail="LM_STUDIO_API_URL is not configured.")

    # This payload structure is a common one for LM Studio, but may need
    # adjustment based on the exact model and server version.
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request.prompt}
        ],
        "temperature": request.temperature,
        "max_tokens": 2048, # Adjust as needed
        "stream": False
    }

    try:
        response = requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions", # This is the typical endpoint
            json=payload
        )
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to LLM API: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "llm_api_endpoint": LM_STUDIO_URL}