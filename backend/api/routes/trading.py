"""
Trading API routes
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from core.database import get_db, Portfolio, Order, User

router = APIRouter()

# Pydantic models
class OrderRequest(BaseModel):
    symbol: str
    order_type: str  # BUY, SELL
    quantity: int
    price: float
    order_side: str = "MARKET"  # MARKET, LIMIT, STOP

class PortfolioResponse(BaseModel):
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    pnl: float
    pnl_percentage: float
    total_value: float

@router.get("/portfolio")
async def get_portfolio(user_id: int = 1, db: Session = Depends(get_db)):
    """Get user portfolio"""
    try:
        portfolio_items = db.query(Portfolio).filter(Portfolio.user_id == user_id).all()
        
        portfolio_data = []
        for item in portfolio_items:
            portfolio_data.append({
                "symbol": item.symbol,
                "quantity": item.quantity,
                "average_price": item.average_price,
                "current_price": item.current_price,
                "pnl": item.pnl,
                "pnl_percentage": item.pnl_percentage,
                "total_value": item.quantity * item.current_price,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat()
            })
            
        return {
            "portfolio": portfolio_data,
            "total_value": sum(item["total_value"] for item in portfolio_data),
            "total_pnl": sum(item["pnl"] for item in portfolio_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio: {str(e)}")

@router.get("/orders")
async def get_orders(user_id: int = 1, db: Session = Depends(get_db)):
    """Get user orders"""
    try:
        orders = db.query(Order).filter(Order.user_id == user_id).order_by(Order.order_time.desc()).all()
        
        orders_data = []
        for order in orders:
            orders_data.append({
                "id": order.id,
                "symbol": order.symbol,
                "order_type": order.order_type,
                "quantity": order.quantity,
                "price": order.price,
                "order_status": order.order_status,
                "order_time": order.order_time.isoformat(),
                "execution_time": order.execution_time.isoformat() if order.execution_time else None
            })
            
        return {"orders": orders_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching orders: {str(e)}")

@router.post("/place-order")
async def place_order(order_request: OrderRequest, user_id: int = 1, db: Session = Depends(get_db)):
    """Place a new order"""
    try:
        # Create new order
        new_order = Order(
            user_id=user_id,
            symbol=order_request.symbol,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            order_status="PENDING"
        )
        
        db.add(new_order)
        db.commit()
        db.refresh(new_order)
        
        # Simulate order execution (in real implementation, this would integrate with broker API)
        if order_request.order_side == "MARKET":
            # Simulate immediate execution
            new_order.order_status = "EXECUTED"
            new_order.execution_time = datetime.utcnow()
            
            # Update portfolio
            await update_portfolio(user_id, order_request, db)
            
        db.commit()
        
        return {
            "order_id": new_order.id,
            "status": "success",
            "message": f"Order placed successfully for {order_request.quantity} shares of {order_request.symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error placing order: {str(e)}")

@router.delete("/cancel-order/{order_id}")
async def cancel_order(order_id: int, user_id: int = 1, db: Session = Depends(get_db)):
    """Cancel an order"""
    try:
        order = db.query(Order).filter(
            Order.id == order_id, 
            Order.user_id == user_id,
            Order.order_status == "PENDING"
        ).first()
        
        if not order:
            raise HTTPException(status_code=404, detail="Order not found or cannot be cancelled")
            
        order.order_status = "CANCELLED"
        db.commit()
        
        return {
            "order_id": order_id,
            "status": "cancelled",
            "message": "Order cancelled successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling order: {str(e)}")

async def update_portfolio(user_id: int, order_request: OrderRequest, db: Session):
    """Update portfolio after order execution"""
    try:
        # Check if position already exists
        existing_position = db.query(Portfolio).filter(
            Portfolio.user_id == user_id,
            Portfolio.symbol == order_request.symbol
        ).first()
        
        if existing_position:
            if order_request.order_type == "BUY":
                # Add to existing position
                total_quantity = existing_position.quantity + order_request.quantity
                total_value = (existing_position.quantity * existing_position.average_price) + \
                             (order_request.quantity * order_request.price)
                new_average_price = total_value / total_quantity
                
                existing_position.quantity = total_quantity
                existing_position.average_price = new_average_price
                existing_position.current_price = order_request.price
                existing_position.updated_at = datetime.utcnow()
            else:
                # Reduce position
                existing_position.quantity -= order_request.quantity
                existing_position.current_price = order_request.price
                existing_position.updated_at = datetime.utcnow()
                
                if existing_position.quantity <= 0:
                    db.delete(existing_position)
        else:
            if order_request.order_type == "BUY":
                # Create new position
                new_position = Portfolio(
                    user_id=user_id,
                    symbol=order_request.symbol,
                    quantity=order_request.quantity,
                    average_price=order_request.price,
                    current_price=order_request.price,
                    pnl=0.0,
                    pnl_percentage=0.0
                )
                db.add(new_position)
                
        db.commit()
        
    except Exception as e:
        print(f"Error updating portfolio: {e}")
        db.rollback()
