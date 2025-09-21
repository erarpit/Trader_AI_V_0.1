"""
Risk Management API routes
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from pydantic import BaseModel
import numpy as np
import pandas as pd

from core.database import get_db, Portfolio, RiskMetrics, User
from services.risk_manager import RiskManager

router = APIRouter()

# Initialize risk manager
risk_manager = RiskManager()

# Pydantic models
class PositionSizeRequest(BaseModel):
    symbol: str
    current_price: float
    risk_percentage: float = 2.0
    stop_loss_percentage: float = 5.0
    portfolio_value: float

class TradeValidationRequest(BaseModel):
    symbol: str
    order_type: str
    quantity: int
    price: float
    user_id: int

class AddPositionRequest(BaseModel):
    symbol: str
    quantity: int
    price: float
    user_id: int

@router.get("/portfolio")
async def get_portfolio_risk(user_id: int = 1, db: Session = Depends(get_db)):
    """Get portfolio risk metrics"""
    try:
        # Get user portfolio
        portfolio_items = db.query(Portfolio).filter(Portfolio.user_id == user_id).all()
        
        if not portfolio_items:
            return {
                "portfolio_value": 0,
                "risk_metrics": {},
                "positions": [],
                "message": "No positions in portfolio"
            }
        
        # Calculate risk metrics
        portfolio_data = []
        total_value = 0
        
        for item in portfolio_items:
            position_value = item.quantity * item.current_price
            total_value += position_value
            
            portfolio_data.append({
                "symbol": item.symbol,
                "quantity": item.quantity,
                "current_price": item.current_price,
                "position_value": position_value,
                "weight": 0,  # Will be calculated below
                "pnl": item.pnl,
                "pnl_percentage": item.pnl_percentage
            })
        
        # Calculate weights
        for position in portfolio_data:
            position["weight"] = position["position_value"] / total_value if total_value > 0 else 0
        
        # Calculate risk metrics
        risk_metrics = risk_manager.calculate_portfolio_risk(portfolio_data, total_value)
        
        # Store risk metrics in database
        risk_record = RiskMetrics(
            user_id=user_id,
            portfolio_value=total_value,
            var_1d=risk_metrics.get("var_1d", 0),
            var_5d=risk_metrics.get("var_5d", 0),
            sharpe_ratio=risk_metrics.get("sharpe_ratio", 0),
            max_drawdown=risk_metrics.get("max_drawdown", 0),
            beta=risk_metrics.get("beta", 0),
            calculated_at=datetime.utcnow()
        )
        
        db.add(risk_record)
        db.commit()
        
        return {
            "portfolio_value": total_value,
            "risk_metrics": risk_metrics,
            "positions": portfolio_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating portfolio risk: {str(e)}")

@router.get("/recommendations")
async def get_risk_recommendations(user_id: int = 1, db: Session = Depends(get_db)):
    """Get risk management recommendations"""
    try:
        # Get portfolio risk data
        portfolio_risk = await get_portfolio_risk(user_id, db)
        
        # Generate recommendations
        recommendations = risk_manager.generate_recommendations(portfolio_risk)
        
        return {
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting risk recommendations: {str(e)}")

@router.post("/position-size")
async def calculate_position_size(request: PositionSizeRequest):
    """Calculate optimal position size based on risk parameters"""
    try:
        position_size = risk_manager.calculate_position_size(
            current_price=request.current_price,
            risk_percentage=request.risk_percentage,
            stop_loss_percentage=request.stop_loss_percentage,
            portfolio_value=request.portfolio_value
        )
        
        return {
            "symbol": request.symbol,
            "recommended_quantity": position_size["quantity"],
            "position_value": position_size["position_value"],
            "risk_amount": position_size["risk_amount"],
            "risk_percentage": request.risk_percentage,
            "stop_loss_price": position_size["stop_loss_price"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating position size: {str(e)}")

@router.post("/validate-trade")
async def validate_trade(request: TradeValidationRequest, db: Session = Depends(get_db)):
    """Validate a trade against risk criteria"""
    try:
        # Get current portfolio
        portfolio_items = db.query(Portfolio).filter(Portfolio.user_id == request.user_id).all()
        
        # Calculate current portfolio value
        current_portfolio_value = sum(item.quantity * item.current_price for item in portfolio_items)
        
        # Calculate trade value
        trade_value = request.quantity * request.price
        
        # Validate trade
        validation_result = risk_manager.validate_trade(
            symbol=request.symbol,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            current_portfolio_value=current_portfolio_value,
            existing_positions=portfolio_items
        )
        
        return {
            "is_valid": validation_result["is_valid"],
            "reasons": validation_result["reasons"],
            "warnings": validation_result.get("warnings", []),
            "trade_value": trade_value,
            "portfolio_impact": validation_result.get("portfolio_impact", 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating trade: {str(e)}")

@router.post("/add-position")
async def add_position_with_risk_management(
    request: AddPositionRequest, 
    db: Session = Depends(get_db)
):
    """Add a position with risk management checks"""
    try:
        # First validate the trade
        validation_request = TradeValidationRequest(
            symbol=request.symbol,
            order_type="BUY",
            quantity=request.quantity,
            price=request.price,
            user_id=request.user_id
        )
        
        validation_result = await validate_trade(validation_request, db)
        
        if not validation_result["is_valid"]:
            return {
                "success": False,
                "message": "Trade validation failed",
                "reasons": validation_result["reasons"],
                "warnings": validation_result.get("warnings", [])
            }
        
        # Add position to portfolio
        existing_position = db.query(Portfolio).filter(
            Portfolio.user_id == request.user_id,
            Portfolio.symbol == request.symbol
        ).first()
        
        if existing_position:
            # Update existing position
            total_quantity = existing_position.quantity + request.quantity
            total_value = (existing_position.quantity * existing_position.average_price) + \
                         (request.quantity * request.price)
            new_average_price = total_value / total_quantity
            
            existing_position.quantity = total_quantity
            existing_position.average_price = new_average_price
            existing_position.current_price = request.price
            existing_position.updated_at = datetime.utcnow()
        else:
            # Create new position
            new_position = Portfolio(
                user_id=request.user_id,
                symbol=request.symbol,
                quantity=request.quantity,
                average_price=request.price,
                current_price=request.price,
                pnl=0.0,
                pnl_percentage=0.0
            )
            db.add(new_position)
        
        db.commit()
        
        return {
            "success": True,
            "message": f"Position added successfully for {request.symbol}",
            "quantity": request.quantity,
            "price": request.price,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding position: {str(e)}")

@router.get("/risk-limits")
async def get_risk_limits():
    """Get current risk limits and thresholds"""
    try:
        limits = risk_manager.get_risk_limits()
        
        return {
            "risk_limits": limits,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting risk limits: {str(e)}")
