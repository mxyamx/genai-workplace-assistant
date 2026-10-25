import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI(title="PSP GenAI Workplace Assistant API", version="1.0")

# Mount static files folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Metrics store
metrics_store = {
    "total_queries": 0,
    "playbooks_generated": 0,
    "feedback_logs": []
}

class ChatRequest(BaseModel):
    prompt: str
    user_role: str = "Employee"

class PlaybookRequest(BaseModel):
    workflow_title: str
    raw_notes: str

class FeedbackRequest(BaseModel):
    feature: str
    rating: int
    comment: str = ""

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/api/chat")
def workplace_chat(req: ChatRequest):
    metrics_store["total_queries"] += 1
    
    if not client:
        return {"response": "[Error] GEMINI_API_KEY is missing from your .env file.", "metrics": metrics_store["total_queries"]}

    models_to_try = ['gemini-3.5-flash', 'gemini-3.5-flash-lite', 'gemini-3.6-flash']

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=req.prompt,
                config={
                    'system_instruction': (
                        "You are an expert internal enterprise AI assistant at an investment firm helping employees navigate digital workplace tools, M365 Copilot, and secure AI adoption. Keep responses professional, clear, and actionable."
                    )
                }
            )
            return {"response": response.text, "metrics": metrics_store["total_queries"]}
        except Exception as e:
            if "503" in str(e):
                time.sleep(0.5)
                continue
            if "404" in str(e):
                continue
            return {"response": f"[API Error]: {str(e)}", "metrics": metrics_store["total_queries"]}

    return {"response": "[Error] Servers busy, please click Submit Query again.", "metrics": metrics_store["total_queries"]}

@app.post("/api/generate-playbook")
def generate_ai_playbook(req: PlaybookRequest):
    metrics_store["playbooks_generated"] += 1

    if not client:
        return {"workflow_title": req.workflow_title, "playbook": "[Error] GEMINI_API_KEY is missing from your .env file."}

    prompt = f"""
    Based on the following raw notes from a team brainstorming session, generate a professional 'AI Playbook' for enterprise adoption.
    
    Workflow Title: {req.workflow_title}
    Raw Notes: {req.raw_notes}
    
    Structure the playbook with these exact markdown sections:
    1. Executive Summary
    2. Step-by-Step AI Integration Workflow
    3. Prompt Templates & Best Practices
    4. Security, Governance & Guardrails
    """
    
    models_to_try = ['gemini-3.5-flash', 'gemini-3.5-flash-lite', 'gemini-3.6-flash']
    
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return {"workflow_title": req.workflow_title, "playbook": response.text}
        except Exception as e:
            if "503" in str(e):
                time.sleep(0.5)
                continue
            if "404" in str(e):
                continue
            return {"workflow_title": req.workflow_title, "playbook": f"[API Error]: {str(e)}"}

    return {"workflow_title": req.workflow_title, "playbook": "[Error] Servers busy, please click Generate again in 3 seconds."}

@app.post("/api/feedback")
def log_feedback(req: FeedbackRequest):
    metrics_store["feedback_logs"].append(req.model_dump() if hasattr(req, 'model_dump') else req.dict())
    return {"status": "success", "total_feedback_received": len(metrics_store["feedback_logs"])}

@app.get("/api/metrics")
def get_metrics():
    return metrics_store

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)