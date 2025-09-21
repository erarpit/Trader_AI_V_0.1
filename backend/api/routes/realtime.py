"""
Real-time data API routes
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio
import json
from datetime import datetime, timedelta

from core.database import get_db, MarketData
from core.nse_api import NSEAPI
from core.bse_api import BSEAPI
from core.websocket_manager import WebSocketManager

router = APIRouter()

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

@router.get("/quote/{symbol}")
async def get_quote(symbol: str, exchange: str = "NSE"):
    """Get live quote for a symbol"""
    try:
        if exchange.upper() == "NSE":
            async with NSEAPI() as nse:
                quote_data = await nse.get_quote(symbol)
        elif exchange.upper() == "BSE":
            async with BSEAPI() as bse:
                quote_data = await bse.get_quote(symbol)
        else:
            raise HTTPException(status_code=400, detail="Invalid exchange. Use NSE or BSE")
            
        if not quote_data:
            raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")
            
        return quote_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching quote: {str(e)}")

@router.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str, 
    exchange: str = "NSE",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get historical data for a symbol"""
    try:
        if exchange.upper() == "NSE":
            async with NSEAPI() as nse:
                historical_data = await nse.get_historical_data(symbol, from_date=from_date, to_date=to_date)
        elif exchange.upper() == "BSE":
            async with BSEAPI() as bse:
                historical_data = await bse.get_historical_data(symbol, from_date=from_date, to_date=to_date)
        else:
            raise HTTPException(status_code=400, detail="Invalid exchange. Use NSE or BSE")
            
        if not historical_data:
            raise HTTPException(status_code=404, detail=f"Historical data not found for {symbol}")
            
        # Store in database for caching
        for data_point in historical_data:
            market_data = MarketData(
                symbol=symbol,
                open_price=data_point.get("open", 0),
                high_price=data_point.get("high", 0),
                low_price=data_point.get("low", 0),
                close_price=data_point.get("close", 0),
                volume=data_point.get("volume", 0),
                timestamp=datetime.now()
            )
            db.add(market_data)
            
        db.commit()
        
        return historical_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching historical data: {str(e)}")

@router.get("/market-status")
async def get_market_status():
    """Get current market status"""
    try:
        nse_status = None
        bse_status = None
        
        # Get NSE status
        async with NSEAPI() as nse:
            nse_status = await nse.get_market_status()
            
        # Get BSE status
        async with BSEAPI() as bse:
            bse_status = await bse.get_market_status()
            
        return {
            "nse": nse_status,
            "bse": bse_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market status: {str(e)}")

@router.get("/top-gainers")
async def get_top_gainers(exchange: str = "NSE"):
    """Get top gaining stocks"""
    try:
        if exchange.upper() == "NSE":
            async with NSEAPI() as nse:
                gainers = await nse.get_top_gainers()
        elif exchange.upper() == "BSE":
            async with BSEAPI() as bse:
                gainers = await bse.get_top_gainers()
        else:
            raise HTTPException(status_code=400, detail="Invalid exchange. Use NSE or BSE")
            
        return gainers or []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching top gainers: {str(e)}")

@router.get("/top-losers")
async def get_top_losers(exchange: str = "NSE"):
    """Get top losing stocks"""
    try:
        if exchange.upper() == "NSE":
            async with NSEAPI() as nse:
                losers = await nse.get_top_losers()
        elif exchange.upper() == "BSE":
            async with BSEAPI() as bse:
                losers = await bse.get_top_losers()
        else:
            raise HTTPException(status_code=400, detail="Invalid exchange. Use NSE or BSE")
            
        return losers or []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching top losers: {str(e)}")

@router.post("/start-feed")
async def start_realtime_feed(symbol: str, exchange: str = "NSE", background_tasks: BackgroundTasks = None):
    """Start real-time feed for a symbol"""
    try:
        # Start background task for real-time updates
        if background_tasks:
            background_tasks.add_task(start_price_feed, symbol, exchange)
            
        return {"message": f"Real-time feed started for {symbol} on {exchange}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting feed: {str(e)}")

@router.post("/stop-feed")
async def stop_realtime_feed(symbol: str):
    """Stop real-time feed for a symbol"""
    try:
        # Implementation to stop the feed
        return {"message": f"Real-time feed stopped for {symbol}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping feed: {str(e)}")

async def start_price_feed(symbol: str, exchange: str):
    """Background task to start price feed"""
    try:
        while True:
            # Get latest quote
            if exchange.upper() == "NSE":
                async with NSEAPI() as nse:
                    quote_data = await nse.get_quote(symbol)
            elif exchange.upper() == "BSE":
                async with BSEAPI() as bse:
                    quote_data = await bse.get_quote(symbol)
            else:
                break
                
            if quote_data:
                # Send to WebSocket subscribers
                await websocket_manager.send_to_subscribers(symbol, quote_data)
                
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"Error in price feed for {symbol}: {e}")
