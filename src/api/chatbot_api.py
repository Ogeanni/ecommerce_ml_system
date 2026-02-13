"""
FastAPI Backend integrating the LangChain/LangGraph Chatbot
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
import os

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import chatbot
from src.chatbot.chatbot_core import EcommerceAgent, create_company_knowledge_base

router = APIRouter()

# Initialize the chatbot agent
print("Initializing E-commerce Agent...")
agent = EcommerceAgent()
print(" Agent initialized successfully!")


# Pydantic models
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    timestamp: str

class SessionInfo(BaseModel):
    session_id: str
    message_count: int


@router.on_event("startup")
async def startup_event():
    """Initialize knowledge base on startup"""
    if not os.path.exists("data/vector/company_vectorstore"):
        print(" Creating company knowledge base...")
        create_company_knowledge_base()
        print(" Knowledge base created!")


@router.get("/")
def home():
    return {
        "message": "OGE E-commerce Company Information Chatbot API",
        "version": "1.0",
        "description": "AI-powered chatbot to answer questions about our e-commerce quality control solutions",
        "endpoints": {
            "chat": "POST /chat - Send a message to the chatbot",
            "sessions": "GET /sessions - List all active sessions",
            "clear_session": "DELETE /session/{session_id} - Clear session history",
        }
    }


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_status": "ready"
    }


@router.post("/chat", response_model = ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    
    Send a message and get AI response based on company information
    """
    try:
        # Get response from agent
        result = agent.chat(user_input=request.message, session_id=request.session_id)
        return ChatResponse(session_id = result["session_id"],
                            response = result["response"],
                            timestamp = result["timestamp"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/sessions")
async def get_sessions():
    """List all active chat sessions"""
    sessions = []
    for session_id, history in agent.message_histories.items():
        sessions.append({
            "session_id": session_id,
            "message_count": len(history.messages),
            "last_activity": datetime.now().isoformat()
        })
    return {
        "sessions": sessions,
        "total": len(sessions)
    }
    

@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a specific session"""
    if session_id not in agent.message_histories:
        return {
            "session_id": session_id,
            "messages": [],
            "total": 0
        }
    history = agent.message_histories[session_id]
    messages = []

    for msg in history.messages:
        messages.append({
            "role": "user" if msg.type == "human" else "assistant",
            "content": msg.content,
            "timestamp": datetime.now().isoformat()
        })
    return {
        "session_id": session_id,
        "messages": messages,
        "total": len(messages)
    }

@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear chat history for a session"""
    if session_id in agent.message_histories:
        agent.clear_history(session_id)
        return{
            "message": f"Session {session_id} cleared successfully",
            "session_id": session_id
        }
    return {
        "message": f"Session {session_id} not found or already empty",
        "session_id": session_id
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.chatbot_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True)