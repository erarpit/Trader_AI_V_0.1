"""
Main FastAPI application for Trader AI
Indian Stock Market Trading Platform
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routers
from api.routes import realtime, trading, ai, risk, monitoring
from core.database import engine, Base
from core.websocket_manager import WebSocketManager

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Trader AI - Indian Stock Market Platform",
    description="Advanced AI-powered trading platform for NSE/BSE markets",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
websocket_manager = WebSocketManager()

# Include routers
app.include_router(realtime.router, prefix="/api/realtime", tags=["Real-time Data"])
app.include_router(trading.router, prefix="/api/trading", tags=["Trading"])
app.include_router(ai.router, prefix="/api/ai", tags=["AI Analysis"])
app.include_router(risk.router, prefix="/api/risk", tags=["Risk Management"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Real-time Monitoring"])

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming WebSocket messages
            await websocket_manager.broadcast(data)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Trader AI is running"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Trader AI - Indian Stock Market Platform",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "True").lower() == "true"
    )
