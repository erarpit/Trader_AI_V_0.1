"""
Main FastAPI application for Trader AI
Indian Stock Market Trading Platform
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
import uvicorn
import os
import json
from datetime import datetime
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

# CORS middleware for AWS deployment
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
cors_credentials = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"
cors_methods = os.getenv("CORS_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(",")
cors_headers = os.getenv("CORS_HEADERS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=cors_credentials,
    allow_methods=cors_methods,
    allow_headers=cors_headers,
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
            try:
                message = json.loads(data)
                # Handle different message types
                if message.get("type") == "subscribe":
                    symbol = message.get("symbol")
                    if symbol:
                        websocket_manager.subscribe(websocket, symbol)
                        await websocket.send_text(json.dumps({
                            "type": "subscription_confirmed",
                            "symbol": symbol,
                            "status": "subscribed"
                        }))
                elif message.get("type") == "unsubscribe":
                    symbol = message.get("symbol")
                    if symbol:
                        websocket_manager.unsubscribe(websocket, symbol)
                        await websocket.send_text(json.dumps({
                            "type": "unsubscription_confirmed",
                            "symbol": symbol,
                            "status": "unsubscribed"
                        }))
                else:
                    # Broadcast other messages
                    await websocket_manager.broadcast(data)
            except json.JSONDecodeError:
                # If not JSON, broadcast as text
                await websocket_manager.broadcast(data)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

# Health check endpoint for AWS load balancer
@app.get("/health")
async def health_check():
    """Comprehensive health check for AWS load balancer"""
    try:
        # Check database connection
        from core.database import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    try:
        # Check Redis connection
        from core.redis_client import redis_client
        await redis_client.connect()
        redis_status = "healthy" if await redis_client.ping() else "unhealthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    overall_status = "healthy" if db_status == "healthy" and redis_status == "healthy" else "unhealthy"
    
    return {
        "status": overall_status,
        "message": "Trader AI is running",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": db_status,
            "redis": redis_status
        }
    }

# Serve React build
FRONTEND_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend_build_resolved"))

# Resolve actual frontend build directory (../frontend/build)
_candidate_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "build"))
if os.path.exists(_candidate_build_dir):
    FRONTEND_BUILD_DIR = _candidate_build_dir

# Mount static assets under /static (from React build)
static_dir = os.path.join(FRONTEND_BUILD_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Root serves React index.html
@app.get("/")
async def serve_root():
    index_path = os.path.join(FRONTEND_BUILD_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {
        "message": "Welcome to Trader AI - Indian Stock Market Platform",
        "version": "1.0.0",
        "docs": "/docs"
    }


# Catch-all to support client-side routing, excluding API and WS paths
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path.startswith("api/") or full_path.startswith("ws") or full_path == "docs" or full_path.startswith("redoc"):
        # Let API/docs/ws be handled by their own routes
        raise HTTPException(status_code=404, detail="Not Found")
    index_path = os.path.join(FRONTEND_BUILD_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend build not found")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "True").lower() == "true"
    )
