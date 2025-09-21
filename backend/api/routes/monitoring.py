"""
Real-time Monitoring API routes
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import logging
from datetime import datetime

from core.database import get_db
from services.realtime_signal_monitor import realtime_monitor

router = APIRouter()

@router.post("/start-monitoring")
async def start_monitoring(symbols: List[str]):
    """Start monitoring symbols for trading signals"""
    try:
        if not symbols:
            raise HTTPException(status_code=400, detail="No symbols provided")
        
        result = await realtime_monitor.start_monitoring(symbols)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting monitoring: {str(e)}")

@router.post("/stop-monitoring")
async def stop_monitoring(symbols: List[str] = None):
    """Stop monitoring symbols"""
    try:
        result = await realtime_monitor.stop_monitoring(symbols)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping monitoring: {str(e)}")

@router.get("/monitoring-status")
async def get_monitoring_status():
    """Get current monitoring status"""
    try:
        status = await realtime_monitor.get_monitoring_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting monitoring status: {str(e)}")

@router.get("/signal-history")
async def get_signal_history(symbol: str = None):
    """Get signal history for symbol or all symbols"""
    try:
        history = await realtime_monitor.get_signal_history(symbol)
        return history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting signal history: {str(e)}")

@router.post("/update-settings")
async def update_monitoring_settings(settings: Dict):
    """Update monitoring settings"""
    try:
        result = await realtime_monitor.update_monitoring_settings(settings)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating settings: {str(e)}")

@router.get("/active-signals")
async def get_active_signals():
    """Get currently active trading signals"""
    try:
        # Get recent signals from all monitored symbols
        all_history = await realtime_monitor.get_signal_history()
        active_signals = []
        
        for symbol, signals in all_history.get('all_signals', {}).items():
            if signals:
                latest_signal = signals[-1]
                # Check if signal is recent (within last hour)
                signal_time = datetime.fromisoformat(latest_signal['timestamp'])
                if datetime.now() - signal_time < timedelta(hours=1):
                    active_signals.append({
                        'symbol': symbol,
                        'signal': latest_signal,
                        'time_ago': str(datetime.now() - signal_time)
                    })
        
        return {
            'active_signals': active_signals,
            'count': len(active_signals),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting active signals: {str(e)}")
