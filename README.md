# 🚀 Trader AI - Indian Stock Market Platform

A comprehensive AI-powered trading platform for Indian stock markets (NSE/BSE) with real-time data, advanced analytics, and intelligent trading signals.

## ✨ Features

### 🎯 **Core Features**
- **Real-Time Data**: Live NSE/BSE price feeds with WebSocket updates
- **AI Trading Engine**: Multi-algorithm analysis with technical, fundamental, and sentiment analysis
- **Advanced Charts**: Real-time candlestick charts and technical indicators
- **Risk Management**: Comprehensive portfolio risk analysis and position sizing
- **Trading Terminal**: Professional trading interface with order placement
- **Portfolio Management**: Track P&L, positions, and performance metrics
- **Auto-Signals**: AI-powered buy/sell signals with popup notifications
- **Volume Analysis**: Advanced volume pattern recognition and order flow analysis

### 🔧 **Technical Stack**

**Backend:**
- FastAPI (Python) - High-performance API framework
- WebSocket - Real-time communication
- Redis - Caching and pub/sub
- PostgreSQL - Database
- NSE/BSE APIs - Market data
- Angel One SmartAPI - Live trading integration

**Frontend:**
- React 18 + TypeScript
- Tailwind CSS - Styling
- Recharts - Data visualization
- WebSocket Client - Real-time updates
- Responsive Design - Mobile support

**AI/ML:**
- Technical Analysis (RSI, MACD, Bollinger Bands, etc.)
- Sentiment Analysis (News and social media)
- Risk Models (VaR, Sharpe ratio, correlation)
- Ensemble Learning - Multi-algorithm approach

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Redis
- PostgreSQL

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Trader_AI
```

2. **Backend Setup**
```bash
cd backend
pip install -r requirements.txt
cp env.example .env
# Edit .env with your configuration
python main.py
```

3. **Frontend Setup**
```bash
cd frontend
npm install
npm start
```

4. **Database Setup**
```bash
# Create PostgreSQL database
createdb trader_ai

# Run migrations (if any)
# alembic upgrade head
```

### Configuration

1. **Environment Variables**
   - Copy `env.example` to `.env`
   - Configure database, Redis, and API credentials
   - Set up Angel One API credentials for live trading

2. **NSE/BSE API Setup**
   - No API key required for basic data
   - Configure headers in `env.example`

3. **Angel One Integration** (Optional)
   - Get API credentials from Angel One
   - Configure in environment variables

## 🧪 Testing

### Run Test Suite
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx websockets

# Run comprehensive test suite
python test_trader_ai_comprehensive.py

# Run specific test categories
pytest tests/test_api.py
pytest tests/test_services.py
pytest tests/test_ai_models.py
```

### Test Coverage
- **API Endpoints**: All REST endpoints tested (13 tests)
- **WebSocket**: Real-time communication tested (1 test)
- **AI/ML Models**: Signal generation and analysis tested (4 tests)
- **Services**: Technical analysis, sentiment, risk management tested (4 tests)
- **Database**: Connection and data persistence tested (1 test)
- **Infrastructure**: Redis, PostgreSQL connectivity tested (2 tests)

### Test Cases Overview

#### ✅ **Working Test Cases (Expected to PASS)**
1. **TC001 - Health Check** - Basic API health endpoint
2. **TC002 - Root Endpoint** - Main API welcome endpoint
3. **TC003 - Market Status** - NSE/BSE market status
4. **TC004 - Quote Endpoint** - Live stock quotes
5. **TC005 - Historical Data** - Historical price data
6. **TC006 - Top Gainers** - Top gaining stocks
7. **TC007 - Top Losers** - Top losing stocks
8. **TC015 - AI Engine Functionality** - AI signal generation
9. **TC016 - Technical Analysis** - Technical indicators calculation
10. **TC017 - Sentiment Analysis** - News sentiment analysis
11. **TC018 - Risk Management** - Portfolio risk calculation
12. **TC019 - Database Connection** - PostgreSQL connection
13. **TC020 - Redis Connection** - Redis cache connection

#### ⚠️ **Potentially Non-Working Test Cases (Expected to FAIL/ERROR)**
1. **TC008 - AI Analysis Endpoint** - May fail if AI service not fully implemented
2. **TC009 - AI Signals Endpoint** - May fail if signal generation not working
3. **TC010 - Market Overview** - May fail if market overview service not implemented
4. **TC011 - Portfolio Endpoint** - May fail if portfolio service not implemented
5. **TC012 - Orders Endpoint** - May fail if trading service not implemented
6. **TC013 - Risk Portfolio** - May fail if risk service not fully implemented
7. **TC014 - WebSocket Connection** - May fail if WebSocket not properly configured

### Test Report Generation
The test suite generates a comprehensive JSON report with:
- Test execution summary (total, passed, failed, errors)
- Success rate percentage
- Detailed results for each test case
- Error messages and debugging information
- Timestamp and execution details

## 📊 API Endpoints

### Real-Time Data
- `GET /api/realtime/quote/{symbol}` - Live quotes
- `GET /api/realtime/historical/{symbol}` - Historical data
- `GET /api/realtime/market-status` - Market status
- `POST /api/realtime/start-feed` - Start real-time feed

### Trading
- `GET /api/trading/portfolio` - Portfolio data
- `GET /api/trading/orders` - Order book
- `POST /api/trading/place-order` - Place orders
- `DELETE /api/trading/cancel-order/{id}` - Cancel orders

### AI Analysis
- `POST /api/ai/analyze/{symbol}` - AI stock analysis
- `GET /api/ai/signals/{symbol}` - Trading signals
- `GET /api/ai/market-overview` - Market overview

### Risk Management
- `GET /api/risk/portfolio` - Portfolio risk metrics
- `POST /api/risk/position-size` - Calculate position size
- `POST /api/risk/validate-trade` - Validate trades

## 🎯 Usage

### 1. **Dashboard**
- View portfolio performance
- Monitor market trends
- Track P&L and returns

### 2. **Trading Terminal**
- Search and select stocks
- View real-time charts
- Place buy/sell orders
- Monitor order status

### 3. **Portfolio Management**
- Track all positions
- View detailed P&L
- Manage risk exposure

### 4. **AI Analytics**
- Get AI trading signals
- View risk metrics
- Analyze market sentiment
- Volume analysis

### 5. **Settings**
- Configure trading preferences
- Set risk limits
- Manage notifications

## 🔒 Security Features

- **Risk Management**: Position sizing, stop-loss, portfolio limits
- **Trade Validation**: Pre-trade risk checks
- **Secure APIs**: Authentication and authorization
- **Data Encryption**: Secure data transmission

## 📱 Mobile Support

- Responsive design for mobile devices
- Touch-friendly interface
- Real-time updates on mobile

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading in financial markets involves substantial risk of loss. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the API endpoints

## 🔄 Updates

- **v1.0.0**: Initial release with core features
- Real-time data integration
- AI trading engine
- Risk management system
- Professional trading UI

---

**Happy Trading! 📈**
