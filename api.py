# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from chatbot import get_chat_response

app = FastAPI(title="Ollama Chatbot API", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    user_id: str                    # Wajib
    session_id: Optional[str] = None
    new_conversation: Optional[bool] = False

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = get_chat_response(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
            new_conversation=request.new_conversation
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"status": "running", "message": "Ollama Chatbot API is ready"}
