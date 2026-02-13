from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import chatbot_api
import model_api

app = FastAPI(title="OGE E-commerce Company Chatbot API")

app.include_router(chatbot_api.router)
app.include_router(model_api.router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def home():
    return {
        "message": "E-commerce Quality Control API",
        "version": "1.0.0",
        "system": "Unified QC System with 4 ML Models",
        "chatbot": "OGE E-commerce Company Information Chatbot API",
        }