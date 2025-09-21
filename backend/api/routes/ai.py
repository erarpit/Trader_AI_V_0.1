"""
AI Analysis API routes
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

from core.database import get_db, AIAnalysis, MarketData
from core.nse_api import NSEAPI
from core.bse_api import BSEAPI
from services.technical_analysis import TechnicalAnalyzer
from services.sentiment_analysis import SentimentAnalyzer
from services.ai_engine import AIEngine
from services.volume_analyzer import VolumeAnalyzer
from services.candlestick_patterns import CandlestickPatternAnalyzer
from services.signal_generator import SignalGenerator
from services.notification_service import NotificationService

router = APIRouter()

# Initialize AI services
technical_analyzer = TechnicalAnalyzer()
sentiment_analyzer = SentimentAnalyzer()
ai_engine = AIEngine()
volume_analyzer = VolumeAnalyzer()
pattern_analyzer = CandlestickPatternAnalyzer()
signal_generator = SignalGenerator()
notification_service = NotificationService()

@router.post("/analyze/{symbol}")
async def analyze_stock(
    symbol: str, 
    exchange: str = "NSE",
    analysis_type: str = "COMPREHENSIVE",
    db: Session = Depends(get_db)
):
    """Perform comprehensive AI analysis on a stock"""
    try:
        # Get historical data
        historical_data = await get_historical_data_for_analysis(symbol, exchange)
        if not historical_data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
            
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        analysis_results = {}
        
        # Technical Analysis
        if analysis_type in ["TECHNICAL", "COMPREHENSIVE"]:
            technical_signals = technical_analyzer.analyze(df)
            analysis_results["technical"] = technical_signals
            
        # Sentiment Analysis
        if analysis_type in ["SENTIMENT", "COMPREHENSIVE"]:
            sentiment_data = await sentiment_analyzer.analyze_stock_sentiment(symbol)
            analysis_results["sentiment"] = sentiment_data
            
        # Volume Analysis
        if analysis_type in ["VOLUME", "COMPREHENSIVE"]:
            volume_data = volume_analyzer.analyze_volume_patterns(df)
            analysis_results["volume"] = volume_data
            
        # Candlestick Pattern Analysis
        if analysis_type in ["PATTERNS", "COMPREHENSIVE"]:
            pattern_data = pattern_analyzer.analyze_patterns(df)
            analysis_results["patterns"] = pattern_data
            
        # AI Engine Analysis
        if analysis_type in ["AI", "COMPREHENSIVE"]:
            ai_signals = ai_engine.generate_signals(df, analysis_results)
            analysis_results["ai_signals"] = ai_signals
            
        # Store analysis in database
        analysis_record = AIAnalysis(
            symbol=symbol,
            analysis_type=analysis_type,
            signal=analysis_results.get("ai_signals", {}).get("signal", "HOLD"),
            confidence=analysis_results.get("ai_signals", {}).get("confidence", 0.0),
            price_target=analysis_results.get("ai_signals", {}).get("price_target", 0.0),
            stop_loss=analysis_results.get("ai_signals", {}).get("stop_loss", 0.0),
            analysis_data=str(analysis_results),
            created_at=datetime.utcnow()
        )
        
        db.add(analysis_record)
        db.commit()
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "results": analysis_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing stock: {str(e)}")

@router.get("/signals/{symbol}")
async def get_trading_signals(
    symbol: str,
    exchange: str = "NSE",
    db: Session = Depends(get_db)
):
    """Get latest trading signals for a symbol"""
    try:
        # Get latest analysis
        latest_analysis = db.query(AIAnalysis).filter(
            AIAnalysis.symbol == symbol
        ).order_by(AIAnalysis.created_at.desc()).first()
        
        if not latest_analysis:
            # Perform fresh analysis
            return await analyze_stock(symbol, exchange, "COMPREHENSIVE", db)
            
        return {
            "symbol": symbol,
            "signal": latest_analysis.signal,
            "confidence": latest_analysis.confidence,
            "price_target": latest_analysis.price_target,
            "stop_loss": latest_analysis.stop_loss,
            "analysis_time": latest_analysis.created_at.isoformat(),
            "analysis_data": eval(latest_analysis.analysis_data) if latest_analysis.analysis_data else {}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting signals: {str(e)}")

@router.get("/market-overview")
async def get_market_overview():
    """Get AI-powered market overview"""
    try:
        # Get top gainers and losers
        async with NSEAPI() as nse:
            gainers = await nse.get_top_gainers()
            losers = await nse.get_top_losers()
            
        # Analyze market sentiment
        market_sentiment = await sentiment_analyzer.analyze_market_sentiment()
        
        return {
            "market_sentiment": market_sentiment,
            "top_gainers": gainers[:10] if gainers else [],
            "top_losers": losers[:10] if losers else [],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting market overview: {str(e)}")

@router.get("/recommendations")
async def get_ai_recommendations(
    limit: int = 10,
    min_confidence: float = 0.7,
    db: Session = Depends(get_db)
):
    """Get AI trading recommendations"""
    try:
        # Get recent high-confidence analyses
        recommendations = db.query(AIAnalysis).filter(
            AIAnalysis.confidence >= min_confidence,
            AIAnalysis.signal.in_(["BUY", "SELL"])
        ).order_by(
            AIAnalysis.confidence.desc(),
            AIAnalysis.created_at.desc()
        ).limit(limit).all()
        
        recommendations_data = []
        for rec in recommendations:
            recommendations_data.append({
                "symbol": rec.symbol,
                "signal": rec.signal,
                "confidence": rec.confidence,
                "price_target": rec.price_target,
                "stop_loss": rec.stop_loss,
                "analysis_time": rec.created_at.isoformat()
            })
            
        return {
            "recommendations": recommendations_data,
            "total_count": len(recommendations_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

async def get_historical_data_for_analysis(symbol: str, exchange: str) -> List[Dict]:
    """Get historical data for analysis"""
    try:
        # Get data from last 6 months
        from_date = (datetime.now() - timedelta(days=180)).strftime("%d-%m-%Y")
        to_date = datetime.now().strftime("%d-%m-%Y")
        
        if exchange.upper() == "NSE":
            async with NSEAPI() as nse:
                return await nse.get_historical_data(symbol, from_date=from_date, to_date=to_date)
        elif exchange.upper() == "BSE":
            async with BSEAPI() as bse:
                return await bse.get_historical_data(symbol, from_date=from_date, to_date=to_date)
        else:
            return []
            
    except Exception as e:
        print(f"Error getting historical data for {symbol}: {e}")
        return []

@router.post("/patterns/{symbol}")
async def analyze_candlestick_patterns(
    symbol: str,
    exchange: str = "NSE",
    db: Session = Depends(get_db)
):
    """Analyze candlestick patterns for a symbol"""
    try:
        # Get historical data
        historical_data = await get_historical_data_for_analysis(symbol, exchange)
        if not historical_data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Analyze patterns
        pattern_analysis = pattern_analyzer.analyze_patterns(df)
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now().isoformat(),
            "pattern_analysis": pattern_analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing patterns: {str(e)}")

@router.post("/generate-signal/{symbol}")
async def generate_trading_signal(
    symbol: str,
    exchange: str = "NSE",
    db: Session = Depends(get_db)
):
    """Generate comprehensive trading signal"""
    try:
        # Get historical data
        historical_data = await get_historical_data_for_analysis(symbol, exchange)
        if not historical_data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Generate signal
        signal_data = await signal_generator.generate_comprehensive_signal(symbol, df, exchange)
        
        if 'error' in signal_data:
            raise HTTPException(status_code=500, detail=signal_data['error'])
        
        return signal_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating signal: {str(e)}")

@router.post("/alert-signal/{symbol}")
async def generate_alert_signal(
    symbol: str,
    exchange: str = "NSE",
    user_phone: str = None,
    db: Session = Depends(get_db)
):
    """Generate alert signal for notifications"""
    try:
        # Get historical data
        historical_data = await get_historical_data_for_analysis(symbol, exchange)
        if not historical_data:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Generate alert signal
        alert_signal = await signal_generator.generate_alert_signal(symbol, df, exchange)
        
        if not alert_signal:
            return {"message": "No alert signal generated", "reason": "Signal doesn't meet alert criteria"}
        
        # Send notification if phone provided
        if user_phone:
            notification_result = await notification_service.send_trading_signal(alert_signal, user_phone)
            alert_signal["notification"] = notification_result
        
        return alert_signal
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating alert signal: {str(e)}")

@router.post("/test-notification")
async def test_notification(phone: str):
    """Test notification setup"""
    try:
        result = await notification_service.test_notification(phone)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending test notification: {str(e)}")

@router.get("/notification-preferences")
async def get_notification_preferences():
    """Get notification preferences"""
    try:
        preferences = notification_service.get_user_preferences()
        return preferences
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting preferences: {str(e)}")

@router.post("/notification-preferences")
async def update_notification_preferences(preferences: dict):
    """Update notification preferences"""
    try:
        notification_service.update_user_preferences(preferences)
        return {"message": "Preferences updated successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating preferences: {str(e)}")
