"""
Database configuration and models
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trader_ai.db")

# Database connection pool settings for AWS RDS
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))

# Create engine with connection pooling for AWS RDS
engine = create_engine(
    DATABASE_URL,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT,
    pool_recycle=DB_POOL_RECYCLE,
    pool_pre_ping=True,  # Verify connections before use
    echo=os.getenv("DEBUG", "False").lower() == "true"
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="user")
    orders = relationship("Order", back_populates="user")

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, index=True)
    quantity = Column(Integer)
    average_price = Column(Float)
    current_price = Column(Float)
    pnl = Column(Float, default=0.0)
    pnl_percentage = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, index=True)
    order_type = Column(String)  # BUY, SELL
    quantity = Column(Integer)
    price = Column(Float)
    order_status = Column(String, default="PENDING")  # PENDING, EXECUTED, CANCELLED
    order_time = Column(DateTime, default=datetime.utcnow)
    execution_time = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="orders")

class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
class AIAnalysis(Base):
    __tablename__ = "ai_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    analysis_type = Column(String)  # TECHNICAL, FUNDAMENTAL, SENTIMENT
    signal = Column(String)  # BUY, SELL, HOLD
    confidence = Column(Float)
    price_target = Column(Float)
    stop_loss = Column(Float)
    analysis_data = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

class RiskMetrics(Base):
    __tablename__ = "risk_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    portfolio_value = Column(Float)
    var_1d = Column(Float)
    var_5d = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    beta = Column(Float)
    calculated_at = Column(DateTime, default=datetime.utcnow)
